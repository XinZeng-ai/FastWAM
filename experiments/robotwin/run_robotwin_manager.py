import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PERSISTENT_WORKER_ENTRY = (
    PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_persistent_worker.py"
)
EVAL_STEP_LIMIT_FILE = (
    PROJECT_ROOT
    / "third_party"
    / "RoboTwin"
    / "task_config"
    / "_eval_step_limit.yml"
)
EVAL_TEST_NUM_FILE = (
    PROJECT_ROOT
    / "third_party"
    / "RoboTwin"
    / "task_config"
    / "_eval_test_num.yml"
)
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    explicit = _resolve_optional_path(
        cfg.EVALUATION.dataset_stats_path,
        base=PROJECT_ROOT,
    )
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    for parent in list(ckpt_path.parents)[:4]:
        candidates.append((parent / "dataset_stats.json").resolve())

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Pass "
        "EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., "
                f"got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., "
                f"got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _load_task_step_limits() -> dict[str, int]:
    if not EVAL_STEP_LIMIT_FILE.exists():
        raise FileNotFoundError(f"Task list file not found: {EVAL_STEP_LIMIT_FILE}")
    with EVAL_STEP_LIMIT_FILE.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {EVAL_STEP_LIMIT_FILE}")

    result: dict[str, int] = {}
    for task_name, step_limit in task_map.items():
        if task_name in result:
            continue
        result[str(task_name)] = int(step_limit)
    return result


def _load_task_episode_counts() -> dict[str, int]:
    if not EVAL_TEST_NUM_FILE.exists():
        raise FileNotFoundError(
            f"Per-task episode count file not found: {EVAL_TEST_NUM_FILE}"
        )
    with EVAL_TEST_NUM_FILE.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid per-task episode map: {EVAL_TEST_NUM_FILE}")

    result: dict[str, int] = {}
    for task_name, episode_count in task_map.items():
        count = int(episode_count)
        if count <= 0:
            raise ValueError(
                f"Episode count must be > 0 for task={task_name}, got {count}"
            )
        result[str(task_name)] = count
    return result


def _parse_success_rate(result_file: Path) -> float:
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    if not math.isfinite(last_value) or not 0.0 <= last_value <= 1.0:
        raise ValueError(
            f"Invalid success rate in {result_file}: {last_value}; "
            "expected a finite value in [0, 1]."
        )
    return last_value


def _phase_result_filename(phase: str) -> str:
    if phase == "clean":
        return "_result_clean.txt"
    if phase == "random":
        return "_result_random.txt"
    raise ValueError(f"Unsupported phase: {phase}")


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _to_jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _select_cuda_device(gpu_id: int, inherited_visible_devices: str | None) -> str:
    if inherited_visible_devices is None:
        return str(gpu_id)
    visible_devices = [
        device.strip()
        for device in inherited_visible_devices.split(",")
        if device.strip()
    ]
    if gpu_id < 0 or gpu_id >= len(visible_devices):
        raise ValueError(
            f"`gpu_id={gpu_id}` is outside inherited CUDA_VISIBLE_DEVICES="
            f"{inherited_visible_devices!r} (available logical GPU IDs: "
            f"0..{len(visible_devices) - 1})."
        )
    return visible_devices[gpu_id]


def _build_assignments(
    tasks: list[str],
    task_step_limits: dict[str, int],
    task_episode_counts: dict[str, int],
    num_workers: int,
    split_phases_across_workers: bool = False,
) -> tuple[list[list[dict[str, Any]]], list[int]]:
    """Greedily balance evaluation work across persistent workers."""
    phase_to_task_config = {
        "clean": "demo_clean",
        "random": "demo_randomized",
    }

    if split_phases_across_workers:
        buckets: list[list[dict[str, Any]]] = [[] for _ in range(num_workers)]
        loads = [0 for _ in range(num_workers)]
        phase_assignments = [
            {
                "task_name": task_name,
                "phase": phase,
                "task_config": phase_to_task_config[phase],
                "eval_num_episodes": task_episode_counts[task_name],
            }
            for task_name in tasks
            for phase in ("clean", "random")
        ]
        phase_assignments.sort(
            key=lambda item: (
                task_step_limits.get(str(item["task_name"]), 1)
                * int(item["eval_num_episodes"])
            ),
            reverse=True,
        )
        for assignment in phase_assignments:
            worker_id = min(range(num_workers), key=lambda idx: loads[idx])
            task_name = str(assignment["task_name"])
            load = (
                task_step_limits.get(task_name, 1)
                * int(assignment["eval_num_episodes"])
            )
            buckets[worker_id].append(assignment)
            loads[worker_id] += load
        return buckets, loads

    task_buckets: list[list[str]] = [[] for _ in range(num_workers)]
    estimated_loads = [0 for _ in range(num_workers)]

    # Longer tasks first gives a substantially better static balance. A task's
    # clean/random phases stay on one worker and reuse that worker's model.
    ordered_tasks = sorted(
        tasks,
        key=lambda task: (
            task_step_limits.get(task, 1) * task_episode_counts[task]
        ),
        reverse=True,
    )
    for task_name in ordered_tasks:
        worker_id = min(
            range(num_workers),
            key=lambda idx: estimated_loads[idx],
        )
        task_buckets[worker_id].append(task_name)
        estimated_loads[worker_id] += (
            2
            * task_step_limits.get(task_name, 1)
            * task_episode_counts[task_name]
        )

    assignments: list[list[dict[str, Any]]] = []
    for task_bucket in task_buckets:
        worker_assignments: list[dict[str, Any]] = []
        for task_name in task_bucket:
            for phase in ("clean", "random"):
                worker_assignments.append(
                    {
                        "task_name": task_name,
                        "phase": phase,
                        "task_config": phase_to_task_config[phase],
                        "eval_num_episodes": task_episode_counts[task_name],
                    }
                )
        assignments.append(worker_assignments)
    return assignments, estimated_loads


def _rebalance_pending_assignments(
    assignments: list[dict[str, Any]],
    task_step_limits: dict[str, int],
    num_workers: int,
    split_phases_across_workers: bool = False,
) -> tuple[list[list[dict[str, Any]]], list[int]]:
    """Balance pending phases, optionally allowing task phases to run concurrently."""
    assignments_by_task: dict[str, list[dict[str, Any]]] = {}
    for assignment in assignments:
        task_name = str(assignment["task_name"])
        assignments_by_task.setdefault(task_name, []).append(assignment)

    def assignment_load(assignment: dict[str, Any]) -> int:
        task_name = str(assignment["task_name"])
        return (
            task_step_limits.get(task_name, 1)
            * int(assignment["eval_num_episodes"])
        )

    buckets: list[list[dict[str, Any]]] = [[] for _ in range(num_workers)]
    loads = [0 for _ in range(num_workers)]
    if split_phases_across_workers:
        ordered_assignments = sorted(
            assignments,
            key=assignment_load,
            reverse=True,
        )
        for assignment in ordered_assignments:
            worker_id = min(range(num_workers), key=lambda idx: loads[idx])
            buckets[worker_id].append(assignment)
            loads[worker_id] += assignment_load(assignment)
        return buckets, loads

    ordered_task_assignments = sorted(
        assignments_by_task.values(),
        key=lambda group: sum(assignment_load(item) for item in group),
        reverse=True,
    )
    for task_assignments in ordered_task_assignments:
        worker_id = min(range(num_workers), key=lambda idx: loads[idx])
        buckets[worker_id].extend(task_assignments)
        loads[worker_id] += sum(
            assignment_load(item) for item in task_assignments
        )
    return buckets, loads


def _ensure_policy_symlink(robotwin_root: Path) -> None:
    source = (
        PROJECT_ROOT / "experiments" / "robotwin" / "fastwam_policy"
    ).resolve()
    target = robotwin_root / "policy" / "fastwam_policy"
    if not source.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {source}")
    if not target.exists() and not target.is_symlink():
        target.symlink_to(source, target_is_directory=True)
        return
    if not target.is_symlink() or target.resolve() != source:
        raise RuntimeError(
            f"Policy path conflict: {target}; expected a symlink to {source}"
        )


@dataclass
class WorkerState:
    worker_id: int
    gpu_id: int
    gpu_worker_id: int
    assignments: list[dict[str, Any]]
    status_file: Path
    process: subprocess.Popen[str]


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="sim_robotwin.yaml",
)
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not PERSISTENT_WORKER_ENTRY.exists():
        raise FileNotFoundError(
            f"Persistent worker entry not found: {PERSISTENT_WORKER_ENTRY}"
        )

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)
    dataset_stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)

    robotwin_root = _resolve_path(
        str(cfg.EVALUATION.robotwin_root),
        base=PROJECT_ROOT,
    )
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")
    _ensure_policy_symlink(robotwin_root)

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    if num_gpus <= 0:
        raise ValueError("`MULTIRUN.num_gpus` must be > 0.")
    workers_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if workers_per_gpu <= 0:
        raise ValueError("`MULTIRUN.max_tasks_per_gpu` must be > 0.")
    num_nodes = int(cfg.MULTIRUN.get("num_nodes", 1))
    node_rank = int(cfg.MULTIRUN.get("node_rank", 0))
    coordination_timeout_sec = int(
        cfg.MULTIRUN.get("coordination_timeout_sec", 3600)
    )
    if num_nodes <= 0:
        raise ValueError("`MULTIRUN.num_nodes` must be > 0.")
    if node_rank < 0 or node_rank >= num_nodes:
        raise ValueError(
            f"`MULTIRUN.node_rank` must be in [0, {num_nodes}), got {node_rank}."
        )
    if coordination_timeout_sec <= 0:
        raise ValueError("`MULTIRUN.coordination_timeout_sec` must be > 0.")
    local_num_workers = num_gpus * workers_per_gpu
    num_workers = num_nodes * local_num_workers
    split_phases_across_workers = bool(
        cfg.MULTIRUN.get("split_phases_across_workers", num_nodes > 1)
    )

    output_dir = _resolve_path(
        str(cfg.EVALUATION.output_dir),
        base=PROJECT_ROOT,
    )
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(
            f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}"
        )
    run_output_dir = (
        PROJECT_ROOT / "evaluate_results" / "robotwin" / ckpt_tag / run_ts
    )
    session_id = str(cfg.MULTIRUN.get("session_id", run_ts)).strip()
    if session_id == "":
        raise ValueError("`MULTIRUN.session_id` must not be empty.")
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in session_id):
        raise ValueError(
            "`MULTIRUN.session_id` may contain only letters, digits, dot, "
            f"underscore, and hyphen; got {session_id!r}."
        )
    resume_existing = bool(cfg.EVALUATION.resume_existing)
    if resume_existing and not run_output_dir.is_dir():
        raise FileNotFoundError(
            "Evaluation resume was requested, but the run directory does not "
            f"exist: {run_output_dir}. Check EVAL_RUN_ID, TASK_CONFIG, and "
            "CKPT_PATH."
        )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    manager_log = run_output_dir / (
        "manager.log" if node_rank == 0 else f"manager.node{node_rank}.log"
    )
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"
    node_status_file = (
        run_output_dir
        / f"manager.{session_id}.node{node_rank}.status.json"
    )

    task_step_limits = _load_task_step_limits()
    task_episode_counts = _load_task_episode_counts()
    task_name_cfg = cfg.EVALUATION.task_name
    if task_name_cfg is None or str(task_name_cfg).strip() == "":
        tasks = list(task_step_limits)
    else:
        tasks = [str(task_name_cfg)]

    missing_episode_counts = [
        task for task in tasks if task not in task_episode_counts
    ]
    if missing_episode_counts:
        raise KeyError(
            "Missing per-task episode counts in "
            f"{EVAL_TEST_NUM_FILE}: {missing_episode_counts}"
        )

    assignments_by_worker, estimated_loads = _build_assignments(
        tasks,
        task_step_limits,
        task_episode_counts,
        num_workers,
        split_phases_across_workers=split_phases_across_workers,
    )
    resumed_phases = 0
    resumed_episodes = 0
    if resume_existing:
        pending_assignments: list[dict[str, Any]] = []
        for assignments in assignments_by_worker:
            for assignment in assignments:
                task_name = str(assignment["task_name"])
                phase = str(assignment["phase"])
                result_file = (
                    run_output_dir
                    / task_name
                    / _phase_result_filename(phase)
                )
                if result_file.exists():
                    # A file is a completion marker only if its final success
                    # rate is parseable and valid. Never silently skip a
                    # malformed or partially written result.
                    _parse_success_rate(result_file)
                    resumed_phases += 1
                    resumed_episodes += int(assignment["eval_num_episodes"])
                    continue
                pending_assignments.append(assignment)
        assignments_by_worker, estimated_loads = (
            _rebalance_pending_assignments(
                pending_assignments,
                task_step_limits,
                num_workers,
                split_phases_across_workers=split_phases_across_workers,
            )
        )
    sim_task = HydraConfig.get().runtime.choices.get("task")
    base_args: dict[str, Any] = {
        "ckpt_setting": str(ckpt_path),
        "seed": int(cfg.seed),
        "policy_name": str(cfg.EVALUATION.policy_name),
        "instruction_type": str(cfg.EVALUATION.instruction_type),
        "sim_cfg_path": str(
            (PROJECT_ROOT / "configs" / "sim_robotwin.yaml").resolve()
        ),
        "sim_task": None if sim_task is None else str(sim_task),
        "mixed_precision": str(cfg.mixed_precision),
        "device": str(cfg.EVALUATION.device),
        "dataset_stats_path": str(dataset_stats_path),
        "rae_stats_dataset": str(cfg.data.rae_stats_dataset),
        "action_horizon": (
            None
            if cfg.EVALUATION.action_horizon is None
            else int(cfg.EVALUATION.action_horizon)
        ),
        "replan_steps": int(cfg.EVALUATION.replan_steps),
        "num_inference_steps": int(cfg.EVALUATION.num_inference_steps),
        "sigma_shift": (
            None
            if cfg.EVALUATION.sigma_shift is None
            else float(cfg.EVALUATION.sigma_shift)
        ),
        "text_cfg_scale": float(cfg.EVALUATION.text_cfg_scale),
        "negative_prompt": str(cfg.EVALUATION.negative_prompt),
        "rand_device": str(cfg.EVALUATION.rand_device),
        "tiled": bool(cfg.EVALUATION.tiled),
        "timing_enabled": bool(cfg.EVALUATION.timing_enabled),
        "skip_get_obs_within_replan": bool(
            cfg.EVALUATION.skip_get_obs_within_replan
        ),
        "save_predicted_video": bool(cfg.EVALUATION.save_predicted_video),
        "predicted_video_native_fps": int(
            cfg.EVALUATION.predicted_video_native_fps
        ),
        "predicted_video_max_episodes": int(
            cfg.EVALUATION.predicted_video_max_episodes
        ),
        "predicted_video_max_replans": int(
            cfg.EVALUATION.predicted_video_max_replans
        ),
    }

    task_rates: dict[str, dict[str, float | None]] = {
        task: {"clean": None, "random": None} for task in tasks
    }
    failed_records: list[dict[str, Any]] = []
    worker_states: list[WorkerState] = []

    def log(message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def write_outputs() -> None:
        clean_mean = _mean_or_none(
            [task_rates[task]["clean"] for task in tasks]
        )
        random_mean = _mean_or_none(
            [task_rates[task]["random"] for task in tasks]
        )
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["task_name", "clean_success_rate", "random_success_rate"]
            )
            for task in tasks:
                writer.writerow(
                    [
                        task,
                        task_rates[task]["clean"],
                        task_rates[task]["random"],
                    ]
                )
            writer.writerow(["__overall__", clean_mean, random_mean])

        payload = {
            "per_task": [
                {
                    "task_name": task,
                    "clean_success_rate": _to_jsonable(
                        task_rates[task]["clean"]
                    ),
                    "random_success_rate": _to_jsonable(
                        task_rates[task]["random"]
                    ),
                }
                for task in tasks
            ],
            "overall": {
                "clean_mean_success_rate": _to_jsonable(clean_mean),
                "random_mean_success_rate": _to_jsonable(random_mean),
            },
        }
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with failed_tasks_file.open("w", encoding="utf-8") as f:
            for record in failed_records:
                f.write(
                    f"{record['task_name']},{record['phase']},"
                    f"gpu={record['gpu_id']},"
                    f"return_code={record['return_code']},"
                    f"reason={record['reason']}\n"
                )

    def terminate_workers() -> None:
        for state in worker_states:
            if state.process.poll() is None:
                log(
                    "terminating persistent worker "
                    f"gpu={state.gpu_id} worker={state.gpu_worker_id}"
                )
                state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in worker_states:
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(
                    "killing persistent worker "
                    f"gpu={state.gpu_id} worker={state.gpu_worker_id}"
                )
                state.process.kill()
                state.process.wait()

    log(
        f"manager start node_rank={node_rank}/{num_nodes} tasks={len(tasks)} "
        f"local_gpu_ids={list(range(num_gpus))} "
        f"workers_per_gpu={workers_per_gpu} "
        f"local_persistent_workers={local_num_workers} "
        f"global_persistent_workers={num_workers} "
        f"assignment_granularity={'phase' if split_phases_across_workers else 'task_pair'} "
        f"episodes_per_phase={sum(task_episode_counts[t] for t in tasks)} "
        f"output_dir={run_output_dir}"
    )
    if resume_existing:
        total_phases = len(tasks) * 2
        log(
            f"resume enabled: skipped completed phases={resumed_phases}/"
            f"{total_phases} completed_episodes={resumed_episodes}; "
            f"pending_phases={total_phases - resumed_phases}"
        )

    _write_json_atomic(
        node_status_file,
        {
            "session_id": session_id,
            "node_rank": node_rank,
            "num_nodes": num_nodes,
            "state": "running",
            "error": None,
        },
    )

    inherited_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    # Interleave global worker buckets across nodes. _build_assignments places
    # the longest tasks into the earliest empty buckets, so contiguous slices
    # would concentrate slow tasks on node 0 when workers outnumber tasks.
    local_worker_ids = list(range(node_rank, num_workers, num_nodes))
    if len(local_worker_ids) != local_num_workers:
        raise RuntimeError(
            "Internal worker partition mismatch: "
            f"node={node_rank}, expected={local_num_workers}, "
            f"actual={len(local_worker_ids)}"
        )
    for local_worker_id, worker_id in enumerate(local_worker_ids):
        assignments = assignments_by_worker[worker_id]
        local_gpu_id = local_worker_id // workers_per_gpu
        global_gpu_id = node_rank * num_gpus + local_gpu_id
        gpu_worker_id = local_worker_id % workers_per_gpu
        if len(assignments) == 0:
            log(
                f"skip node={node_rank} local_gpu={local_gpu_id} "
                f"worker={gpu_worker_id}: "
                "no assigned tasks"
            )
            continue

        selected_device = _select_cuda_device(
            local_gpu_id,
            inherited_visible_devices,
        )
        worker_label = f"gpu{global_gpu_id}_worker{gpu_worker_id}"
        status_file = run_output_dir / f"worker_{worker_label}_status.json"
        job_file = run_output_dir / f"worker_{worker_label}_job.json"
        job_payload = {
            "worker_id": worker_id,
            "node_rank": node_rank,
            "gpu_id": global_gpu_id,
            "local_gpu_id": local_gpu_id,
            "gpu_worker_id": gpu_worker_id,
            "robotwin_root": str(robotwin_root),
            "run_output_dir": str(run_output_dir),
            "status_file": str(status_file),
            "base_args": base_args,
            "assignments": assignments,
        }
        job_file.write_text(
            json.dumps(job_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = selected_device
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            sys.executable,
            "-u",
            str(PERSISTENT_WORKER_ENTRY),
            "--job-file",
            str(job_file),
        ]
        log(
            f"launch persistent worker node={node_rank} "
            f"global_gpu={global_gpu_id} local_gpu={local_gpu_id} "
            f"worker={gpu_worker_id} "
            f"physical_device={selected_device} "
            f"tasks={len({a['task_name'] for a in assignments})} "
            f"phases={len(assignments)} "
            f"estimated_steps={estimated_loads[worker_id]}"
        )
        process = subprocess.Popen(
            command,
            cwd=str(robotwin_root),
            env=env,
            text=True,
        )
        worker_states.append(
            WorkerState(
                worker_id=worker_id,
                gpu_id=global_gpu_id,
                gpu_worker_id=gpu_worker_id,
                assignments=assignments,
                status_file=status_file,
                process=process,
            )
        )

    has_failure = False
    failure_message = ""
    running_workers = list(worker_states)
    try:
        while running_workers:
            progressed = False
            for state in list(running_workers):
                return_code = state.process.poll()
                if return_code is None:
                    continue
                progressed = True
                running_workers.remove(state)
                if return_code != 0:
                    has_failure = True
                    current_task = "<worker_startup>"
                    current_phase = "unknown"
                    error_text = ""
                    if state.status_file.exists():
                        worker_status = json.loads(
                            state.status_file.read_text(encoding="utf-8")
                        )
                        current_task = (
                            worker_status.get("current_task") or current_task
                        )
                        current_phase = (
                            worker_status.get("current_phase") or current_phase
                        )
                        error_text = str(worker_status.get("error") or "")
                    failure_message = (
                        f"persistent worker failed: gpu={state.gpu_id}, "
                        f"worker={state.gpu_worker_id}, "
                        f"task={current_task}, phase={current_phase}, "
                        f"return_code={return_code}"
                    )
                    failed_records.append(
                        {
                            "task_name": current_task,
                            "phase": current_phase,
                            "gpu_id": state.gpu_id,
                            "return_code": return_code,
                            "reason": "persistent_worker_failed",
                        }
                    )
                    log(failure_message)
                    if error_text:
                        log(
                            f"worker gpu={state.gpu_id} "
                            f"worker={state.gpu_worker_id} traceback saved in "
                            f"{state.status_file}"
                        )
                    terminate_workers()
                    running_workers.clear()
                    break
                log(
                    f"persistent worker gpu={state.gpu_id} "
                    f"worker={state.gpu_worker_id} finished"
                )
            if has_failure:
                break
            if not progressed:
                time.sleep(POLL_INTERVAL_SEC)
    except KeyboardInterrupt:
        log("manager interrupted; terminating persistent workers")
        terminate_workers()
        _write_json_atomic(
            node_status_file,
            {
                "session_id": session_id,
                "node_rank": node_rank,
                "num_nodes": num_nodes,
                "state": "failed",
                "error": "manager_interrupted",
                "failed_records": failed_records,
            },
        )
        raise

    _write_json_atomic(
        node_status_file,
        {
            "session_id": session_id,
            "node_rank": node_rank,
            "num_nodes": num_nodes,
            "state": "failed" if has_failure else "completed",
            "error": failure_message if has_failure else None,
            "failed_records": failed_records,
        },
    )

    if num_nodes > 1 and node_rank != 0:
        if has_failure:
            raise RuntimeError(failure_message)
        log(
            f"node {node_rank} finished successfully; "
            "rank 0 will aggregate the shared results"
        )
        return

    if num_nodes > 1:
        log(
            f"rank 0 waiting for {num_nodes} node completion markers "
            f"for session={session_id}"
        )
        deadline = time.time() + coordination_timeout_sec
        node_payloads: dict[int, dict[str, Any]] = {}
        while len(node_payloads) < num_nodes:
            for expected_rank in range(num_nodes):
                if expected_rank in node_payloads:
                    continue
                path = (
                    run_output_dir
                    / f"manager.{session_id}.node{expected_rank}.status.json"
                )
                if not path.exists():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if payload.get("session_id") != session_id:
                    continue
                if payload.get("state") not in {"completed", "failed"}:
                    continue
                node_payloads[expected_rank] = payload
                log(
                    f"node completion marker received: node={expected_rank} "
                    f"state={payload.get('state')}"
                )
            if len(node_payloads) == num_nodes:
                break
            if time.time() >= deadline:
                missing_nodes = sorted(set(range(num_nodes)) - set(node_payloads))
                has_failure = True
                failure_message = (
                    "timed out waiting for multi-node evaluation completion: "
                    f"missing_nodes={missing_nodes}, session={session_id}"
                )
                failed_records.append(
                    {
                        "task_name": "<multi_node_coordination>",
                        "phase": "unknown",
                        "gpu_id": -1,
                        "return_code": -1,
                        "reason": failure_message,
                    }
                )
                log(failure_message)
                break
            time.sleep(POLL_INTERVAL_SEC)

        for expected_rank, payload in node_payloads.items():
            if expected_rank == node_rank:
                continue
            for record in payload.get("failed_records", []):
                failed_records.append(record)
            if payload.get("state") == "failed":
                has_failure = True
                if failure_message == "":
                    failure_message = str(
                        payload.get("error")
                        or f"evaluation node {expected_rank} failed"
                    )

    # Recover every result that reached disk, including phases skipped during
    # resume and work from workers terminated after another GPU failed.
    failed_keys = {
        (record["task_name"], record["phase"]) for record in failed_records
    }
    for task_name in tasks:
        for phase in ("clean", "random"):
            result_file = (
                run_output_dir
                / task_name
                / _phase_result_filename(phase)
            )
            if result_file.exists():
                try:
                    success_rate = _parse_success_rate(result_file)
                    task_rates[task_name][phase] = success_rate
                    log(
                        f"done task={task_name} phase={phase} "
                        f"success_rate={success_rate:.4f}"
                    )
                except Exception as exc:
                    if (task_name, phase) not in failed_keys:
                        failed_records.append(
                            {
                                "task_name": task_name,
                                "phase": phase,
                                "gpu_id": -1,
                                "return_code": 0,
                                "reason": f"result_parse_failed:{exc!r}",
                            }
                        )
                        if not has_failure:
                            has_failure = True
                            failure_message = (
                                f"result parse failed: task={task_name}, "
                                f"phase={phase}, gpu={state.gpu_id}, "
                                f"error={exc!r}"
                            )
            elif (task_name, phase) not in failed_keys:
                failed_records.append(
                    {
                        "task_name": task_name,
                        "phase": phase,
                        "gpu_id": -1,
                        "return_code": -1,
                        "reason": (
                            "aborted_not_completed"
                            if has_failure
                            else "result_missing"
                        ),
                    }
                )
                if not has_failure:
                    has_failure = True
                    failure_message = (
                        f"result missing: task={task_name}, phase={phase}, "
                        f"file={result_file}"
                    )

    write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")
    if has_failure:
        raise RuntimeError(failure_message)
    log("manager finished successfully")


if __name__ == "__main__":
    main()

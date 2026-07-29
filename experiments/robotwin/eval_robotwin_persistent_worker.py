"""Persistent RoboTwin evaluation worker.

One process is bound to one GPU, loads FastWAM once, and evaluates all assigned
task/phase pairs with the same model instance.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


class _Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(
            bool(getattr(stream, "isatty", lambda: False)())
            for stream in self.streams
        )


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    job_file = args.job_file.expanduser().resolve()
    job = json.loads(job_file.read_text(encoding="utf-8"))

    worker_id = int(job["worker_id"])
    gpu_id = int(job["gpu_id"])
    gpu_worker_id = int(job["gpu_worker_id"])
    robotwin_root = Path(job["robotwin_root"]).expanduser().resolve()
    status_file = Path(job["status_file"]).expanduser().resolve()
    run_output_dir = Path(job["run_output_dir"]).expanduser().resolve()
    assignments = list(job["assignments"])
    base_args = dict(job["base_args"])

    worker_label = f"gpu{gpu_id}_worker{gpu_worker_id}"
    worker_log = run_output_dir / f"worker_{worker_label}.log"
    log_stream = worker_log.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_stream)
    sys.stderr = _Tee(sys.__stderr__, log_stream)

    os.chdir(robotwin_root)
    for path in (
        robotwin_root,
        robotwin_root / "script",
        robotwin_root / "policy",
        robotwin_root / "description" / "utils",
    ):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    # Import only after the manager has restricted CUDA_VISIBLE_DEVICES for
    # this process, so torch sees exactly one physical GPU.
    import eval_policy as robotwin_eval
    from test_render import Sapien_TEST

    status: dict[str, Any] = {
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "gpu_worker_id": gpu_worker_id,
        "state": "starting",
        "current_task": None,
        "current_phase": None,
        "current_eval_num_episodes": None,
        "completed": [],
        "error": None,
    }
    _write_status(status_file, status)

    print(
        f"[persistent-worker {worker_label}] start "
        f"assignments={len(assignments)} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}",
        flush=True,
    )

    model = None
    try:
        # Preserve the original RoboTwin renderer preflight, but run it only
        # once for the lifetime of this GPU worker.
        Sapien_TEST()

        for assignment in assignments:
            task_name = str(assignment["task_name"])
            phase = str(assignment["phase"])
            task_config = str(assignment["task_config"])
            eval_num_episodes = int(assignment["eval_num_episodes"])

            status["state"] = "running"
            status["current_task"] = task_name
            status["current_phase"] = phase
            status["current_eval_num_episodes"] = eval_num_episodes
            _write_status(status_file, status)

            eval_args = dict(base_args)
            eval_args["task_name"] = task_name
            eval_args["task_config"] = task_config
            eval_args["eval_num_episodes"] = eval_num_episodes
            eval_args["eval_output_dir"] = str(run_output_dir / task_name)

            load_mode = "load model" if model is None else "reuse model"
            print(
                f"[persistent-worker {worker_label}] task={task_name} "
                f"phase={phase} episodes={eval_num_episodes} ({load_mode})",
                flush=True,
            )
            model = robotwin_eval.main(eval_args, model=model)

            status["completed"].append(
                {
                    "task_name": task_name,
                    "phase": phase,
                }
            )
            status["current_task"] = None
            status["current_phase"] = None
            _write_status(status_file, status)

        status["state"] = "completed"
        _write_status(status_file, status)
        print(f"[persistent-worker {worker_label}] finished", flush=True)
    except BaseException:
        status["state"] = "failed"
        status["error"] = traceback.format_exc()
        _write_status(status_file, status)
        raise


if __name__ == "__main__":
    main()

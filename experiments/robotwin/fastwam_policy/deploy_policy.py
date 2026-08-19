import logging
import os
import sys
import time
import inspect
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.video_io import save_mp4

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
    rae_stats_dataset: Optional[str] = None,
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")
    if not _is_none_like(rae_stats_dataset):
        stats_dataset = str(rae_stats_dataset)
        overrides.extend(
            [
                f"data.rae_stats_dataset={stats_dataset}",
                f"data.rae_stats_filename={stats_dataset}_stats.pt",
            ]
        )

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        array = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(frame)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
            array = np.transpose(array, (1, 2, 0))
        if np.issubdtype(array.dtype, np.floating):
            if float(array.min()) >= 0.0 and float(array.max()) <= 1.01:
                array = array * 255.0
            elif float(array.min()) >= -1.01 and float(array.max()) <= 1.01:
                array = (array + 1.0) * 127.5
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB frame [H,W,3], got {array.shape}")
    return np.ascontiguousarray(array)


def _write_rgb_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if len(frames) == 0:
        raise ValueError(f"Cannot save an empty video: {path}")
    if fps <= 0:
        raise ValueError(f"`fps` must be positive, got {fps}")

    normalized = [_frame_to_rgb_array(frame) for frame in frames]
    height, width = normalized[0].shape[:2]
    for frame in normalized:
        if frame.shape[:2] != (height, width):
            raise ValueError(
                f"Video frame size mismatch: expected {(height, width)}, got {frame.shape[:2]}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        # Reuse the same imageio/FFMPEG writer as training validation.
        save_mp4(
            [Image.fromarray(frame, mode="RGB") for frame in normalized],
            str(tmp_path),
            fps=fps,
        )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _make_comparison_frames(
    predicted_frames: list[np.ndarray],
    rollout_frames: list[np.ndarray],
    step_offsets: Optional[list[int]] = None,
) -> list[np.ndarray]:
    frame_count = min(len(rollout_frames), len(predicted_frames))
    if step_offsets is None:
        step_offsets = list(range(frame_count))
    if len(step_offsets) < frame_count:
        raise ValueError(
            f"`step_offsets` has {len(step_offsets)} entries for {frame_count} frames."
        )
    comparison: list[np.ndarray] = []
    for frame_idx in range(frame_count):
        step_offset = int(step_offsets[frame_idx])
        rollout = Image.fromarray(_frame_to_rgb_array(rollout_frames[frame_idx]))
        predicted = Image.fromarray(_frame_to_rgb_array(predicted_frames[frame_idx]))
        if rollout.size != predicted.size:
            rollout = rollout.resize(predicted.size, resample=Image.BILINEAR)
        canvas = Image.new("RGB", (predicted.width * 2, predicted.height))
        canvas.paste(predicted, (0, 0))
        canvas.paste(rollout, (predicted.width, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, predicted.width, 20), fill=(0, 0, 0))
        draw.rectangle(
            (predicted.width, 0, predicted.width * 2, 20),
            fill=(0, 0, 0),
        )
        draw.text((5, 4), f"model prediction | step {step_offset}", fill=(255, 255, 255))
        draw.text(
            (predicted.width + 5, 4),
            f"rollout observation | step {step_offset}",
            fill=(255, 255, 255),
        )
        comparison.append(np.asarray(canvas, dtype=np.uint8))
    return comparison


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        action_video_freq_ratio: int,
        save_predicted_video: bool,
        predicted_video_native_fps: int,
        predicted_video_max_episodes: int,
        predicted_video_max_replans: int,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self._action_video_freq_ratio = int(action_video_freq_ratio)
        self.save_predicted_video = bool(save_predicted_video)
        self.predicted_video_native_fps = int(predicted_video_native_fps)
        self.predicted_video_max_episodes = int(predicted_video_max_episodes)
        self.predicted_video_max_replans = int(predicted_video_max_replans)

        if self._action_video_freq_ratio <= 0:
            raise ValueError(
                "`action_video_freq_ratio` must be positive, got "
                f"{self._action_video_freq_ratio}"
            )
        if self.predicted_video_native_fps <= 0:
            raise ValueError(
                "`predicted_video_native_fps` must be positive, got "
                f"{self.predicted_video_native_fps}"
            )
        if self.predicted_video_max_episodes < 0:
            raise ValueError("`predicted_video_max_episodes` must be >= 0.")
        if self.predicted_video_max_replans < 0:
            raise ValueError("`predicted_video_max_replans` must be >= 0.")
        if self.save_predicted_video:
            supports_joint_video = (
                hasattr(self.model, "infer_joint")
                and "num_video_frames"
                in inspect.signature(self.model.infer_action).parameters
            )
            if not supports_joint_video:
                raise ValueError(
                    "Predicted-video saving requires a joint/IDM model whose "
                    "`infer_action` accepts `num_video_frames`; the selected model "
                    f"is {type(self.model).__name__}."
                )

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = -1
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
        self._prediction_output_root: Optional[Path] = None
        self._prediction_phase: Optional[str] = None
        self._prediction_replan_index = 0
        self._active_prediction: Optional[dict[str, Any]] = None
        self._episode_prediction: Optional[dict[str, Any]] = None

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | "
            "horizon=%d | replan=%d | save_predicted_video=%s",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
            self.save_predicted_video,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_rgb(self, observation: Dict[str, Any]) -> np.ndarray:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        return np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        image = self._build_robotwin_rgb(observation)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def configure_evaluation_output(
        self,
        eval_output_dir: str,
        task_config: str,
    ) -> None:
        """Set task/phase output context; called once before every eval phase."""
        self._finalize_active_prediction()
        self._flush_episode_prediction()
        phase = "randomized" if "randomized" in str(task_config).lower() else "clean"
        self._prediction_phase = phase
        self._prediction_output_root = (
            Path(eval_output_dir).expanduser().resolve()
            / "predicted_videos"
            / phase
        )
        self.episode_count = -1
        self._prediction_replan_index = 0

    def _should_save_current_prediction(self) -> bool:
        if not self.save_predicted_video or self._prediction_output_root is None:
            return False
        if (
            self.predicted_video_max_episodes > 0
            and self.episode_count >= self.predicted_video_max_episodes
        ):
            return False
        if (
            self.predicted_video_max_replans > 0
            and self._prediction_replan_index >= self.predicted_video_max_replans
        ):
            return False
        return True

    def _save_prediction_clip(
        self,
        predicted_video: list[Any],
        observation: Dict[str, Any],
    ) -> None:
        if self._prediction_output_root is None:
            return

        native_frames = [_frame_to_rgb_array(frame) for frame in predicted_video]
        expected_native_frames = (
            self.action_horizon // self._action_video_freq_ratio + 1
        )
        if len(native_frames) != expected_native_frames:
            logger.warning(
                "Predicted video/action timeline mismatch: native=%d ratio=%d "
                "expected_native=%d action_horizon=%d",
                len(native_frames),
                self._action_video_freq_ratio,
                expected_native_frames,
                self.action_horizon,
            )

        if (
            self._episode_prediction is None
            or self._episode_prediction["episode_index"] != self.episode_count
        ):
            self._episode_prediction = {
                "episode_index": self.episode_count,
                "phase": self._prediction_phase,
                "comparison_frames": [],
                "replans": [],
            }

        episode_prediction = self._episode_prediction
        replan_record = {
            "replan_index": self._prediction_replan_index,
            "native_model_frames": len(native_frames),
            "native_action_step_offsets": [
                idx * self._action_video_freq_ratio
                for idx in range(len(native_frames))
            ],
            "action_horizon": self.action_horizon,
            "executed_replan_steps": self.replan_steps,
            "comparison_frames": 0,
            "comparison_episode_video_frame_range": None,
        }
        episode_prediction["replans"].append(replan_record)

        self._active_prediction = {
            "native_frames": native_frames,
            "max_rollout_frames": (
                (len(native_frames) - 1) * self._action_video_freq_ratio + 1
            ),
            "rollout_frames": [self._build_robotwin_rgb(observation)],
            "replan_record": replan_record,
        }

    def _append_rollout_observation(self, observation: Dict[str, Any]) -> None:
        if self._active_prediction is None:
            return
        rollout_frames = self._active_prediction["rollout_frames"]
        max_rollout_frames = int(self._active_prediction["max_rollout_frames"])
        if len(rollout_frames) < max_rollout_frames:
            rollout_frames.append(self._build_robotwin_rgb(observation))

    def _finalize_active_prediction(self, next_replan_starts: bool = False) -> None:
        active = self._active_prediction
        self._active_prediction = None
        if active is None:
            return
        try:
            rollout_frames = active["rollout_frames"]
            native_frames = active["native_frames"]
            native_step_offsets = list(
                range(
                    0,
                    len(rollout_frames),
                    self._action_video_freq_ratio,
                )
            )
            if next_replan_starts:
                # The next chunk starts from the real observation at
                # `replan_steps`. Keep only this chunk's strictly earlier
                # timestamps so the episode timeline neither runs past the
                # executed actions nor duplicates the boundary observation.
                native_step_offsets = [
                    offset
                    for offset in native_step_offsets
                    if offset < self.replan_steps
                ]
            native_rollout_frames = [
                rollout_frames[offset] for offset in native_step_offsets
            ]
            native_comparison = _make_comparison_frames(
                native_frames,
                native_rollout_frames,
                step_offsets=native_step_offsets,
            )
            if self._episode_prediction is None:
                raise RuntimeError(
                    "Missing episode prediction accumulator while finalizing replan."
                )
            comparison_start = len(
                self._episode_prediction["comparison_frames"]
            )
            self._episode_prediction["comparison_frames"].extend(
                native_comparison
            )
            replan_record = active["replan_record"]
            replan_record["comparison_frames"] = len(native_comparison)
            replan_record["next_replan_starts"] = next_replan_starts
            replan_record["comparison_action_step_offsets"] = (
                native_step_offsets[: len(native_comparison)]
            )
            replan_record["comparison_episode_video_frame_range"] = [
                comparison_start,
                comparison_start + len(native_comparison),
            ]
        except Exception:
            logger.exception(
                "Failed to finalize predicted-video comparison; continuing evaluation."
            )

    def _flush_episode_prediction(self) -> None:
        episode_prediction = self._episode_prediction
        self._episode_prediction = None
        if episode_prediction is None or self._prediction_output_root is None:
            return

        try:
            episode_index = int(episode_prediction["episode_index"])
            phase = str(episode_prediction["phase"])
            comparison_frames = episode_prediction["comparison_frames"]
            output_prefix = (
                self._prediction_output_root
                / f"episode_{episode_index:03d}_{phase}"
            )
            comparison_path = output_prefix.with_name(
                f"{output_prefix.name}_comparison_executed_timeline.mp4"
            )

            if len(comparison_frames) > 0:
                _write_rgb_video(
                    comparison_path,
                    comparison_frames,
                    fps=self.predicted_video_native_fps,
                )

            metadata = {
                "phase": phase,
                "episode_index": episode_index,
                "native_frames_per_replan": self._num_video_frames,
                "comparison_frames_per_full_nonfinal_replan": (
                    (self.replan_steps - 1) // self._action_video_freq_ratio + 1
                ),
                "saved_replans": len(episode_prediction["replans"]),
                "comparison_total_frames": len(comparison_frames),
                "native_fps": self.predicted_video_native_fps,
                "temporal_interpolation": False,
                "comparison_video": comparison_path.name,
                "comparison_resolution": {
                    "width": (
                        int(comparison_frames[0].shape[1])
                        if comparison_frames
                        else None
                    ),
                    "height": (
                        int(comparison_frames[0].shape[0])
                        if comparison_frames
                        else None
                    ),
                },
                "comparison_note": (
                    "Executed replan segments are concatenated without duplicate "
                    "boundary frames. Left is the model prediction; right is the "
                    "model-input rollout observation before the aligned action."
                ),
                "rollout_alignment": (
                    "Native prediction frame i aligns to the rollout observation "
                    "before action offset i*action_video_freq_ratio. For every "
                    "nonfinal replan, only offsets strictly below replan_steps "
                    "are kept; the next chunk supplies the boundary observation."
                ),
                "action_video_freq_ratio": self._action_video_freq_ratio,
                "replans": episode_prediction["replans"],
            }
            metadata_path = output_prefix.with_name(
                f"{output_prefix.name}_predicted_video_metadata.json"
            )
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info(
                "Saved episode-level prediction comparison | episode=%d phase=%s "
                "replans=%d comparison_frames=%d dir=%s",
                episode_index,
                phase,
                len(episode_prediction["replans"]),
                len(comparison_frames),
                self._prediction_output_root,
            )
        except Exception:
            logger.exception(
                "Failed to save episode-level prediction comparison; "
                "continuing evaluation."
            )

    def finalize_evaluation_output(self) -> None:
        """Flush the final episode when a clean/randomized phase finishes."""
        self._finalize_active_prediction()
        self._flush_episode_prediction()

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        save_this_prediction = self._should_save_current_prediction()
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            if save_this_prediction:
                infer_joint_kwargs = {
                    **infer_kwargs,
                    "test_action_with_infer_action": False,
                }
                pred = self.model.infer_joint(**infer_joint_kwargs)
            else:
                pred = self.model.infer_action(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        if save_this_prediction:
            try:
                predicted_video = pred.get("video")
                if predicted_video is None:
                    video_latents = pred.get("video_latents")
                    if not isinstance(video_latents, torch.Tensor):
                        raise KeyError(
                            "Inference returned neither RGB `video` nor tensor `video_latents`."
                        )
                    if not bool(
                        getattr(self.model.vae, "supports_pca_visualization", False)
                    ):
                        raise RuntimeError(
                            "Decoder-free visual tokenizer provides no PCA visualization."
                        )
                    pca_video = self.model.vae.decode_pca(
                        video_latents,
                        output_size=(image_tensor.shape[-2], image_tensor.shape[-1]),
                        seed=self.seed,
                    )[0]
                    pca_video = (
                        (pca_video.detach().float().cpu().clamp(-1, 1) + 1.0)
                        * 127.5
                    ).round().to(torch.uint8)
                    predicted_video = [
                        Image.fromarray(
                            pca_video[:, frame_index].permute(1, 2, 0).numpy()
                        )
                        for frame_index in range(pca_video.shape[1])
                    ]
                    logger.info(
                        "Saving V-JEPA prediction as PCA pseudo-colour; this is not an RGB reconstruction."
                    )
                self._save_prediction_clip(predicted_video, observation)
            except Exception:
                logger.exception(
                    "Failed to save predicted video for episode=%d replan=%d; "
                    "continuing evaluation.",
                    self.episode_count,
                    self._prediction_replan_index,
                )
                self._active_prediction = None
        self._prediction_replan_index += 1

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            if self._active_prediction is not None:
                self._append_rollout_observation(observation)
                self._finalize_active_prediction(next_replan_starts=True)
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)
        elif observation is not None:
            self._append_rollout_observation(observation)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self._finalize_active_prediction()
        self._flush_episode_prediction()
        self.pending_actions.clear()
        self.episode_count += 1
        self._prediction_replan_index = 0
        self.step_count = 0
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
        rae_stats_dataset=usr_args.get("rae_stats_dataset"),
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    save_predicted_video = _parse_bool(
        usr_args.get(
            "save_predicted_video",
            cfg.EVALUATION.get("save_predicted_video", False),
        )
    )
    predicted_video_native_fps = int(
        usr_args.get(
            "predicted_video_native_fps",
            cfg.EVALUATION.get("predicted_video_native_fps", 8),
        )
    )
    predicted_video_max_episodes = int(
        usr_args.get(
            "predicted_video_max_episodes",
            cfg.EVALUATION.get("predicted_video_max_episodes", 1),
        )
    )
    predicted_video_max_replans = int(
        usr_args.get(
            "predicted_video_max_replans",
            cfg.EVALUATION.get("predicted_video_max_replans", 0),
        )
    )
    skip_get_obs_within_replan = _parse_bool(
        usr_args.get(
            "skip_get_obs_within_replan",
            cfg.EVALUATION.get("skip_get_obs_within_replan", False),
        )
    )
    if (
        save_predicted_video and skip_get_obs_within_replan
    ):
        raise ValueError(
            "Predicted-video rollout comparison requires "
            "`skip_get_obs_within_replan=false` so every executed action offset "
            "has an aligned observation."
        )
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        action_video_freq_ratio=action_video_freq_ratio,
        save_predicted_video=save_predicted_video,
        predicted_video_native_fps=predicted_video_native_fps,
        predicted_video_max_episodes=predicted_video_max_episodes,
        predicted_video_max_replans=predicted_video_max_replans,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()

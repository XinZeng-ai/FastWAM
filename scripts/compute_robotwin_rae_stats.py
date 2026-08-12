#!/usr/bin/env python
"""Compute position-wise RAE latent statistics on the RoboTwin train split."""

from __future__ import annotations

import argparse
import itertools
import math
import os
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from fastwam.models.wan22.rae_video_tokenizer import (
    _DINOv3K7Encoder,
    _MAEEncoder,
    _SigLIP2Encoder,
)


class _RoboTwinEpisodeVideoDataset(Dataset):
    """Decode each selected episode once, then reuse the training video preprocessing."""

    def __init__(self, robot_dataset):
        self.robot_dataset = robot_dataset
        self.sources = robot_dataset.lerobot_dataset.multi_dataset._datasets
        self.episodes = []
        for source_index, source in enumerate(self.sources):
            source_episode_ids = source.episodes
            if source_episode_ids is None:
                source_episode_ids = list(range(source.num_episodes))
            for episode_position, episode_id in enumerate(source_episode_ids):
                self.episodes.append(
                    (source_index, episode_position, int(episode_id))
                )

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        source_index, episode_position, episode_id = self.episodes[index]
        source = self.sources[source_index]
        start = int(source.episode_data_index["from"][episode_position])
        stop = int(source.episode_data_index["to"][episode_position])
        selected = source.hf_dataset[list(range(start, stop))]
        timestamps = [
            float(value.item() if isinstance(value, torch.Tensor) else value)
            for value in selected["timestamp"]
        ]
        query_timestamps = {
            key: timestamps for key in source.meta.video_keys
        }
        decoded = source._query_videos(query_timestamps, episode_id)
        camera_keys = [
            meta["lerobot_key"]
            for meta in self.robot_dataset.lerobot_dataset.image_meta
        ]
        raw_video = torch.stack([decoded[key] for key in camera_keys], dim=0)
        video, _ = self.robot_dataset.prepare_video(
            raw_video, temporal_indices=slice(None)
        )
        return {"video": video, "episode_index": episode_id}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation",
        required=True,
        choices=("dinov3_k7", "siglip2_b", "mae_b"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--data-config",
        default="robotwin",
        choices=("robotwin", "robotwin_clean2500"),
        help="RoboTwin data config whose training split defines the stats population.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--hf-cache-dir",
        default=".cache/huggingface",
        help=(
            "Writable Hugging Face datasets cache. Relative paths are resolved "
            "from the FastWAM root."
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Debug-only cap. Omit for formal training-split statistics.",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=None,
        help=(
            "Deterministically sample this many frames through uniformly spaced "
            "training-window indices. At least 100000 frames are required for a "
            "sampled stats file to be eligible for formal RoboTwin training."
        ),
    )
    parser.add_argument(
        "--sample-phase",
        type=float,
        default=0.0,
        help=(
            "Fractional offset in [0, 1) within each uniform sampling interval. "
            "The default 0 preserves the original deterministic sample; use 0.5 "
            "to compute an interleaved diagnostic sample."
        ),
    )
    parser.add_argument(
        "--unique-frames",
        action="store_true",
        help=(
            "Cover every source frame in the selected training episodes exactly "
            "once. Non-overlapping nine-frame windows are selected per episode "
            "and episode-tail padding is excluded from the moments."
        ),
    )
    parser.add_argument(
        "--episode-streaming",
        action="store_true",
        help=(
            "Decode every selected episode once and encode it in frame chunks. "
            "Requires --unique-frames and avoids repeated random seeks. In this "
            "mode --batch-size is the encoder frame chunk size."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Use every visible CUDA GPU through torch.nn.DataParallel.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sample_frames is not None and args.sample_frames <= 0:
        raise ValueError(f"`--sample-frames` must be positive, got {args.sample_frames}")
    if args.unique_frames and args.sample_frames is not None:
        raise ValueError("`--unique-frames` and `--sample-frames` are mutually exclusive")
    if args.episode_streaming and not args.unique_frames:
        raise ValueError("`--episode-streaming` requires `--unique-frames`")
    if not 0.0 <= args.sample_phase < 1.0:
        raise ValueError(f"`--sample-phase` must be in [0, 1), got {args.sample_phase}")
    root = Path(__file__).resolve().parents[1]
    hf_cache_dir = Path(args.hf_cache_dir).expanduser()
    if not hf_cache_dir.is_absolute():
        hf_cache_dir = root / hf_cache_dir
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    # RoboTwin is loaded through datasets.load_dataset(). Some managed training
    # environments have a read-only /root, so never rely on its default cache.
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir / "datasets")

    task = f"robotwin_uncond_3cam_384_rae_{args.representation}_1e-4"
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"task={task}", f"data={args.data_config}"],
        )
    # Stats are position-wise over independent frames; they have no temporal
    # axis. Decode nine contiguous frames instead of the training action window
    # of 33 frames followed by stride-4 selection. The camera mosaic, spatial
    # resize, pixel normalization, and resulting [3,9,384,320] tensor are
    # unchanged, while video decoding work is reduced by roughly 33/9.
    cfg.data.train.num_frames = 9
    cfg.data.train.action_video_freq_ratio = 1
    cfg.data.train.processor.num_obs_steps = 9
    tokenizer_cfg = OmegaConf.to_container(
        cfg.model.vision_tokenizer_config, resolve=True
    )
    encoder_path = Path(tokenizer_cfg["encoder_path"])
    if not encoder_path.is_absolute():
        encoder_path = root / encoder_path

    if args.representation == "dinov3_k7":
        repo_path = Path(tokenizer_cfg["dinov3_repo_path"])
        if not repo_path.is_absolute():
            repo_path = root / repo_path
        encoder = _DINOv3K7Encoder(encoder_path, repo_path)
        channels = 1024
    elif args.representation == "siglip2_b":
        encoder = _SigLIP2Encoder(encoder_path)
        channels = 768
    else:
        encoder = _MAEEncoder(encoder_path)
        channels = 768

    device = torch.device(args.device)
    encoder.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    visible_cuda_devices = 0
    if device.type == "cuda":
        visible_cuda_devices = torch.cuda.device_count()
    if args.data_parallel:
        if device.type != "cuda":
            raise ValueError("`--data-parallel` requires --device cuda")
        if visible_cuda_devices < 2:
            raise RuntimeError(
                "`--data-parallel` requested but fewer than two CUDA devices "
                f"are visible (count={visible_cuda_devices})"
            )
        encoder = torch.nn.DataParallel(
            encoder, device_ids=list(range(visible_cuda_devices))
        )
        print(f"Using DataParallel across {visible_cuda_devices} CUDA devices")
    dataset = instantiate(cfg.data.train)
    selected_episode_count = len(dataset.lerobot_dataset.episode_data_index["from"])
    full_dataset_length = len(dataset)
    frames_per_sample = 9
    selected_windows = full_dataset_length
    sampling_method = "complete_train_windows"
    expected_unique_frames = None
    if args.unique_frames:
        episode_from = dataset.lerobot_dataset.episode_data_index["from"].tolist()
        episode_to = dataset.lerobot_dataset.episode_data_index["to"].tolist()
        indices = []
        expected_unique_frames = 0
        for start, stop in zip(episode_from, episode_to, strict=True):
            start = int(start)
            stop = int(stop)
            if stop <= start:
                raise RuntimeError(
                    f"Invalid RoboTwin episode bounds: from={start}, to={stop}"
                )
            indices.extend(range(start, stop, frames_per_sample))
            expected_unique_frames += stop - start
        selected_windows = len(indices)
        if args.episode_streaming:
            dataset = _RoboTwinEpisodeVideoDataset(dataset)
            selected_windows = len(dataset)
            sampling_method = "complete_train_unique_frames_episode_streaming"
        else:
            dataset = Subset(dataset, indices)
            sampling_method = "complete_train_unique_frames"
        print(
            "Complete RoboTwin unique-frame coverage: "
            f"frames={expected_unique_frames}, "
            f"decode_units={selected_windows}, "
            f"training_episodes={selected_episode_count}"
        )
    elif args.sample_frames is not None:
        selected_windows = min(
            math.ceil(args.sample_frames / frames_per_sample),
            full_dataset_length,
        )
        if selected_windows < 1:
            raise RuntimeError("Uniform stats sampling selected no RoboTwin windows")
        if selected_windows == 1:
            indices = [0]
        elif args.sample_phase == 0.0:
            indices = (
                torch.linspace(
                    0,
                    full_dataset_length - 1,
                    steps=selected_windows,
                    dtype=torch.float64,
                )
                .round()
                .to(torch.int64)
                .tolist()
            )
        else:
            positions = torch.arange(selected_windows, dtype=torch.float64)
            positions = (positions + args.sample_phase) / selected_windows
            indices = (positions * full_dataset_length).floor().to(torch.int64).tolist()
            indices[-1] = min(indices[-1], full_dataset_length - 1)
        if len(set(indices)) != len(indices):
            raise RuntimeError(
                "Uniform RoboTwin stats indices contain duplicates; reduce "
                f"--sample-frames (dataset windows={full_dataset_length})"
            )
        dataset = Subset(dataset, indices)
        sampling_method = "uniform_dataset_window_indices"
        print(
            "Uniform RoboTwin stats sampling: "
            f"requested_frames={args.sample_frames}, "
            f"selected_windows={selected_windows}/{full_dataset_length}"
        )
    loader_kwargs = {
        "dataset": dataset,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.episode_streaming:
        loader_kwargs["batch_size"] = None
    else:
        loader_kwargs["batch_size"] = args.batch_size
    loader = DataLoader(**loader_kwargs)

    total = torch.zeros(channels, 24, 20, dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    count = 0
    batches = loader
    if args.max_batches is not None:
        batches = itertools.islice(loader, args.max_batches)
    with torch.no_grad():
        for sample in tqdm(batches, desc="RoboTwin RAE stats"):
            video = sample["video"]
            if args.episode_streaming:
                if tuple(video.shape[:1] + video.shape[2:]) != (3, 384, 320):
                    raise ValueError(
                        "Expected streamed RoboTwin episode [3,T,384,320], got "
                        f"{tuple(video.shape)}"
                    )
                images = video.permute(1, 0, 2, 3)
                for image_chunk in images.split(args.batch_size, dim=0):
                    tokens = encoder(
                        image_chunk.to(device=device, non_blocking=True)
                    )
                    latents = tokens.transpose(1, 2).reshape(
                        image_chunk.shape[0], channels, 24, 20
                    )
                    latents = latents.double()
                    total += latents.sum(dim=0).cpu()
                    total_sq += latents.square().sum(dim=0).cpu()
                    count += latents.shape[0]
                continue
            if tuple(video.shape[1:]) != (3, 9, 384, 320):
                raise ValueError(
                    "Expected RoboTwin mosaic [B,3,9,384,320], got "
                    f"{tuple(video.shape)}"
                )
            batch, _, frames, height, width = video.shape
            images = video.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, 3, height, width
            )
            tokens = encoder(images.to(device=device, non_blocking=True))
            latents = tokens.transpose(1, 2).reshape(
                batch * frames, channels, 24, 20
            )
            if args.unique_frames:
                image_is_pad = sample.get("image_is_pad")
                if image_is_pad is None:
                    raise KeyError(
                        "RoboTwin sample has no `image_is_pad`; cannot exclude "
                        "episode-tail padding in --unique-frames mode"
                    )
                valid_frames = ~image_is_pad.reshape(-1).bool()
                if valid_frames.numel() != latents.shape[0]:
                    raise ValueError(
                        "Flattened image padding mask does not match RAE frames: "
                        f"mask={valid_frames.numel()}, latents={latents.shape[0]}"
                    )
                latents = latents[valid_frames.to(device=latents.device)]
            if args.sample_frames is not None:
                remaining = args.sample_frames - count
                if remaining <= 0:
                    break
                latents = latents[:remaining]
            latents = latents.double()
            total += latents.sum(dim=0).cpu()
            total_sq += latents.square().sum(dim=0).cpu()
            count += latents.shape[0]

    if count < 2:
        raise RuntimeError(f"Need at least two frames for statistics, got {count}")
    if (
        args.unique_frames
        and args.max_batches is None
        and count != expected_unique_frames
    ):
        raise RuntimeError(
            "Unique-frame coverage count mismatch: "
            f"expected={expected_unique_frames}, accumulated={count}"
        )
    mean = total / count
    var = (total_sq / count - mean.square()).clamp_min(0.0)
    complete_train_split = args.sample_frames is None and args.max_batches is None
    formal_eligible = args.max_batches is None and (
        complete_train_split
        or (
            args.sample_frames is not None
            and args.sample_frames >= 100_000
            and count == args.sample_frames
        )
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": mean.float(),
            "var": var.float(),
            "count": count,
            "dataset": (
                "robotwin_clean2500_train"
                if args.data_config == "robotwin_clean2500"
                else "robotwin_train"
            ),
            "data_config": args.data_config,
            "selected_train_episodes": selected_episode_count,
            "episode_selection": OmegaConf.to_container(
                cfg.data.train.get("episode_selection", {}), resolve=True
            ),
            "representation": args.representation,
            "grid_size": (24, 20),
            "input_size": (384, 320),
            "complete_train_split": complete_train_split,
            "formal_eligible": formal_eligible,
            "sampling_method": sampling_method,
            "unique_frames": args.unique_frames,
            "episode_streaming": args.episode_streaming,
            "expected_unique_frames": expected_unique_frames,
            "full_dataset_windows": full_dataset_length,
            "selected_windows": selected_windows,
            "requested_frames": args.sample_frames,
            "sample_phase": args.sample_phase,
            "stats_cuda_devices": visible_cuda_devices,
            "stats_dataset_num_frames": 9,
            "stats_action_video_freq_ratio": 1,
            "max_batches": args.max_batches,
        },
        output,
    )
    print(f"Saved {args.representation} RoboTwin train stats ({count} frames) to {output}")
    if args.max_batches is not None:
        print("WARNING: --max-batches was set; these debug statistics are not for formal training.")
    elif not formal_eligible:
        print(
            "WARNING: sampled stats contain fewer than 100000 frames and are "
            "not eligible for formal RoboTwin training."
        )


if __name__ == "__main__":
    main()

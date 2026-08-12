#!/usr/bin/env python
"""Compute action/proprio normalization stats on the clean-2500 train split."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data/robotwin2.0/dataset_stats_clean2500_train.json",
    )
    parser.add_argument("--hf-cache-dir", default=".cache/huggingface")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    hf_cache = Path(args.hf_cache_dir).expanduser()
    if not hf_cache.is_absolute():
        hf_cache = root / hf_cache
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache / "datasets")
    # Import after setting the cache variables; datasets resolves its cache
    # constants at import time.
    from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
    from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=robotwin_uncond_3cam_384_1e-4",
                "data=robotwin_clean2500",
            ],
        )
    data_cfg = cfg.data.train
    processor = instantiate(data_cfg.processor)
    selection = OmegaConf.to_container(data_cfg.episode_selection, resolve=True)
    dataset = BaseLerobotDataset(
        dataset_dirs=list(data_cfg.dataset_dirs),
        shape_meta=OmegaConf.to_container(data_cfg.shape_meta, resolve=True),
        obs_size=int(data_cfg.num_frames),
        action_size=int(data_cfg.num_frames) - 1,
        val_set_proportion=float(data_cfg.val_set_proportion),
        is_training_set=True,
        global_sample_stride=int(data_cfg.global_sample_stride),
        verify_episode_files=bool(data_cfg.verify_episode_files),
        episode_selection=selection,
    )
    selected_episodes = len(dataset.episode_data_index["from"])
    if selected_episodes != 2475:
        raise RuntimeError(
            f"Expected 2475 clean training episodes after the 1% split, got {selected_episodes}."
        )

    stats = dataset.get_dataset_stats(processor)
    stats["dataset"] = "robotwin_clean2500_train"
    stats["source_clean_episodes"] = 2500
    stats["selected_train_episodes"] = selected_episodes
    stats["val_set_proportion"] = float(data_cfg.val_set_proportion)
    stats["split_seed"] = 42
    stats["episode_selection"] = selection

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_stats_to_json(stats, str(output))
    print(f"Saved clean-2500 train normalization stats to {output}")


if __name__ == "__main__":
    main()

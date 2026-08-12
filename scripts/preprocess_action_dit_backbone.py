import argparse
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.action_dit_init import build_action_backbone_from_video
from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}. Expected one of: float32, float16, bfloat16.")


def _parse_bool(name: str) -> bool:
    value = str(name).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {name}")


def _is_unresolved_interpolation(value: Any) -> bool:
    return isinstance(value, str) and "${" in value and "}" in value


def _resolve_from_video_cfg(value: Any, video_cfg: dict[str, Any]) -> Any:
    if not _is_unresolved_interpolation(value):
        return value
    text = str(value).strip()
    if not (text.startswith("${") and text.endswith("}")):
        return value
    expr = text[2:-1]
    if not expr.startswith("video_dit_config."):
        return value
    key = expr.split(".", 1)[1]
    if key not in video_cfg:
        return value
    resolved = video_cfg[key]
    return value if _is_unresolved_interpolation(resolved) else resolved


def _load_model_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = OmegaConf.load(str(path))
    if "video_dit_config" not in cfg or "action_dit_config" not in cfg:
        raise ValueError(
            f"`{path}` must contain both `video_dit_config` and `action_dit_config` at top level."
        )

    video_cfg = OmegaConf.to_container(cfg.video_dit_config, resolve=False)
    action_cfg = OmegaConf.to_container(cfg.action_dit_config, resolve=False)
    if not isinstance(video_cfg, dict) or not isinstance(action_cfg, dict):
        raise ValueError("`video_dit_config` and `action_dit_config` must resolve to dicts.")

    if _is_unresolved_interpolation(video_cfg.get("action_dim")):
        print("[WARN] `video_dit_config.action_dim` is unresolved; defaulting to 7 for preprocessing.")
        video_cfg["action_dim"] = 7

    if _is_unresolved_interpolation(action_cfg.get("action_dim")):
        print("[WARN] `action_dit_config.action_dim` is unresolved; defaulting to 7 for preprocessing.")
        action_cfg["action_dim"] = 7

    for key in ["num_heads", "attn_head_dim", "num_layers", "text_dim", "freq_dim"]:
        action_cfg[key] = _resolve_from_video_cfg(action_cfg.get(key), video_cfg)

    return video_cfg, action_cfg, cfg


def _require_int_config(cfg: dict[str, Any], key: str) -> int:
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return int(value)


def _require_float_config(cfg: dict[str, Any], key: str) -> float:
    value = cfg.get(key)
    if _is_unresolved_interpolation(value):
        raise ValueError(f"`{key}` is unresolved interpolation: {value}")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess ActionDiT backbone weights from WanVideoDiT and save as .pt payload."
    )
    parser.add_argument("--model-config", required=True, help="Path to model yaml, e.g. configs/model/fastwam.yaml")
    parser.add_argument("--output", required=True, help="Output .pt path for preprocessed ActionDiT backbone.")
    parser.add_argument("--device", default="cpu", help="Device for loading model and preprocessing.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--apply-alpha-scaling",
        default="true",
        help="Whether to apply alpha=sqrt(dv/da) when the last dimension is resized (true/false). Default: true.",
    )
    args = parser.parse_args()

    model_config_path = Path(args.model_config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    apply_alpha_scaling = _parse_bool(args.apply_alpha_scaling)

    video_cfg, action_cfg, cfg = _load_model_config(model_config_path)
    torch_dtype = _parse_dtype(args.dtype)
    redirect_common_files = _parse_bool(cfg.get("redirect_common_files", False))

    int_fields = ["hidden_dim", "action_dim", "ffn_dim", "num_layers", "num_heads", "attn_head_dim", "text_dim", "freq_dim"]
    for key in int_fields:
        action_cfg[key] = _require_int_config(action_cfg, key)
    action_cfg["eps"] = _require_float_config(action_cfg, "eps")

    print(f"[INFO] Loaded model config from {model_config_path}. "
          f"Preprocessing ActionDiT backbone with dtype={torch_dtype} on device={args.device}, "
          f"apply_alpha_scaling={apply_alpha_scaling}.")
    components = load_wan22_ti2v_5b_components(
        device=args.device,
        torch_dtype=torch_dtype,
        model_id=cfg.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"),
        tokenizer_model_id=cfg.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
        redirect_common_files=redirect_common_files,
        dit_config=video_cfg,
    )
    video_expert = components.dit

    action_expert = ActionDiT(**action_cfg).to(device=args.device, dtype=torch_dtype)
    if int(action_cfg["num_heads"]) != int(video_expert.num_heads):
        raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
    if int(action_cfg["attn_head_dim"]) != int(video_expert.attn_head_dim):
        raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
    if int(action_cfg["num_layers"]) != int(len(video_expert.blocks)):
        raise ValueError("ActionDiT `num_layers` must match video expert.")

    action_state = action_expert.state_dict()
    video_state = video_expert.state_dict()
    backbone_keys = ActionDiT.backbone_key_set(action_state.keys())

    backbone_state_dict, init_report = build_action_backbone_from_video(
        video_state=video_state,
        action_state=action_state,
        backbone_keys=backbone_keys,
        apply_alpha_scaling=apply_alpha_scaling,
    )
    backbone_state_dict = {
        key: value.detach().to(device="cpu").contiguous()
        for key, value in backbone_state_dict.items()
    }

    payload = {
        "policy": {
            "skip_prefixes": list(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES),
            "alpha_scaling": bool(apply_alpha_scaling),
            "interpolation": "sequential_1d_linear_align_corners_true",
        },
        "backbone_state_dict": backbone_state_dict,
        "meta": {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        },
    }
    torch.save(payload, str(output_path))

    skipped = len(action_state) - len(backbone_keys)
    print(
        "[INFO] Saved ActionDiT backbone payload to "
        f"{output_path} (copied={init_report['copied']}, "
        f"interpolated={init_report['interpolated']}, skipped={skipped})."
    )


if __name__ == "__main__":
    main()

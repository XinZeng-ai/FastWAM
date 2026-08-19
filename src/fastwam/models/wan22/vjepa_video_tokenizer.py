"""Frozen V-JEPA 2.1 causal-pair visual tokenizer for FastWAM.

Each latent time step encodes [previous, current] through one video tubelet;
the first observation is represented by [first, first].

V-JEPA has no pixel decoder.  ``decode_pca`` is intentionally a diagnostic
pseudo-colour visualization and is never used by training or evaluation.
"""

from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .rae_video_tokenizer import _require_dir, _require_file

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_VJEPA_ENCODER_TYPE = "vjepa2_1_vitg_causal_pair"


def _install_timm_drop_path_fallback() -> None:
    """Provide the sole timm symbol used by the official V-JEPA encoder.

    The FastWAM environment does not otherwise depend on timm. V-JEPA imports
    only ``timm.models.layers.drop_path`` and inference has drop probability
    zero, so keeping this tiny canonical implementation avoids introducing an
    unpinned package dependency.
    """
    try:
        from timm.models.layers import drop_path as _  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1.0 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    timm_module = ModuleType("timm")
    models_module = ModuleType("timm.models")
    layers_module = ModuleType("timm.models.layers")
    layers_module.drop_path = drop_path
    models_module.layers = layers_module
    timm_module.models = models_module
    sys.modules.setdefault("timm", timm_module)
    sys.modules.setdefault("timm.models", models_module)
    sys.modules.setdefault("timm.models.layers", layers_module)


def _normalize_vjepa_input(
    images: torch.Tensor,
    mean: Iterable[float] = _IMAGENET_MEAN,
    std: Iterable[float] = _IMAGENET_STD,
) -> torch.Tensor:
    """Convert FastWAM [-1, 1] pixels to official ImageNet normalization."""
    images = (images + 1.0) * 0.5
    shape = (1, 3) + (1,) * (images.ndim - 2)
    mean_t = images.new_tensor(tuple(mean)).view(shape)
    std_t = images.new_tensor(tuple(std)).view(shape)
    return (images - mean_t) / std_t


class VJEPA21ViTGEncoder(nn.Module):
    """Official V-JEPA 2.1 ViT-G/16-2B target encoder, loaded locally."""

    hidden_size = 1664
    patch_size = 16
    tubelet_size = 2

    def __init__(self, checkpoint_path: Path, repo_path: Path):
        super().__init__()
        expected_source = repo_path / "app" / "vjepa_2_1" / "models" / "vision_transformer.py"
        if not expected_source.is_file():
            raise FileNotFoundError(
                "V-JEPA 2 source checkout is invalid: expected "
                f"{expected_source}. Set VJEPA2_REPO_PATH to the official "
                "facebookresearch/vjepa2 checkout."
            )

        repo_str = str(repo_path)
        inserted = repo_str not in sys.path
        if inserted:
            sys.path.insert(0, repo_str)
        try:
            _install_timm_drop_path_fallback()
            from app.vjepa_2_1.models.vision_transformer import vit_gigantic_xformers

            # Construct on meta: random initialization of a 2B fp32 model is
            # both unnecessary and slow because every parameter is replaced
            # by the local target-encoder checkpoint below.
            # The official constructor calls .item() on its stochastic-depth
            # schedule. Keep that tiny schedule on CPU while all parameters
            # are created on meta.
            original_linspace = torch.linspace

            def _cpu_linspace(*args, **kwargs):
                kwargs.setdefault("device", "cpu")
                return original_linspace(*args, **kwargs)

            torch.linspace = _cpu_linspace
            try:
                with torch.device("meta"):
                    self.model = vit_gigantic_xformers(
                        patch_size=16,
                        img_size=(384, 384),
                        num_frames=64,
                        tubelet_size=2,
                        use_sdpa=True,
                        use_SiLU=False,
                        wide_SiLU=True,
                        uniform_power=False,
                        use_rope=True,
                        img_temporal_dim_size=1,
                        interpolate_rope=True,
                    )
            finally:
                torch.linspace = original_linspace
        finally:
            if inserted:
                sys.path.remove(repo_str)

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        if not isinstance(checkpoint, dict) or "target_encoder" not in checkpoint:
            raise ValueError(
                f"V-JEPA checkpoint has no target_encoder state: {checkpoint_path}"
            )
        raw_state = checkpoint["target_encoder"]
        if not isinstance(raw_state, dict):
            raise TypeError(
                "V-JEPA target_encoder must be a state dict, got "
                f"{type(raw_state)} in {checkpoint_path}"
            )
        prefix = "module.backbone."
        invalid_keys = [key for key in raw_state if not key.startswith(prefix)]
        if invalid_keys:
            raise RuntimeError(
                "Unexpected V-JEPA target_encoder key prefix; first invalid keys: "
                f"{invalid_keys[:8]}"
            )
        state = {key[len(prefix):]: value for key, value in raw_state.items()}
        expected = self.model.state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        mismatched = {
            key: (tuple(state[key].shape), tuple(expected[key].shape))
            for key in set(state) & set(expected)
            if tuple(state[key].shape) != tuple(expected[key].shape)
        }
        if missing or unexpected or mismatched:
            raise RuntimeError(
                f"Incompatible V-JEPA target encoder {checkpoint_path}: "
                f"missing={missing}, unexpected={unexpected}, "
                f"shape_mismatch={mismatched}"
            )
        # assign=True replaces the randomly initialized 2B parameter storages
        # instead of holding a second full fp32 copy while loading.
        self.model.load_state_dict(state, strict=True, assign=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1:3] != (3, 2):
            raise ValueError(
                "V-JEPA causal-pair encoder expects [N,3,2,H,W], got "
                f"{tuple(inputs.shape)}"
            )
        inputs = _normalize_vjepa_input(inputs)
        autocast = (
            torch.autocast("cuda", dtype=inputs.dtype)
            if inputs.is_cuda and inputs.dtype in {torch.float16, torch.bfloat16}
            else nullcontext()
        )
        with autocast:
            return self.model(inputs, training=False)


def build_causal_pairs(video: torch.Tensor) -> torch.Tensor:
    """Build [previous,current] clips, using [first,first] at time zero."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"Expected video [B,3,T,H,W], got {tuple(video.shape)}")
    previous = torch.cat((video[:, :, :1], video[:, :, :-1]), dim=2)
    return torch.stack((previous, video), dim=3)


@torch.no_grad()
def pca_visualize_latents(
    latents: torch.Tensor,
    output_size: tuple[int, int] = (384, 320),
    seed: int = 0,
) -> torch.Tensor:
    """Map [B,C,T,h,w] patch latents to PCA pseudo-colour in [-1,1]."""
    if latents.ndim != 5 or latents.shape[1] < 3:
        raise ValueError(f"PCA visualization expects [B,C,T,H,W], got {tuple(latents.shape)}")
    batch, channels, frames, grid_h, grid_w = latents.shape
    samples = latents.permute(0, 2, 3, 4, 1).reshape(-1, channels).float()
    centered = samples - samples.mean(dim=0, keepdim=True)
    devices = [latents.device.index] if latents.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        _, _, basis = torch.pca_lowrank(centered, q=3, center=False)
    projected = centered @ basis[:, :3]
    max_indices = projected.abs().argmax(dim=0)
    signs = torch.sign(
        projected[max_indices, torch.arange(3, device=projected.device)]
    )
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    projected = projected * signs
    low = torch.quantile(projected, 0.01, dim=0)
    high = torch.quantile(projected, 0.99, dim=0)
    rgb = ((projected - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    rgb = rgb.reshape(batch, frames, grid_h, grid_w, 3).permute(0, 1, 4, 2, 3)
    rgb = F.interpolate(
        rgb.reshape(batch * frames, 3, grid_h, grid_w),
        size=tuple(output_size),
        mode="nearest",
    )
    return rgb.reshape(batch, frames, 3, *output_size).permute(0, 2, 1, 3, 4) * 2.0 - 1.0


class VJEPA21VideoTokenizer(nn.Module):
    """Wan-VAE-compatible latent encoder without a learned RGB decoder."""

    temporal_downsample_factor = 1
    upsampling_factor = 16
    supports_rgb_decode = False
    supports_pca_visualization = True

    def __init__(
        self,
        encoder_type: str,
        encoder_path: str,
        stats_path: str,
        stats_dataset: str = "robotwin_clean2500_train",
        vjepa2_repo_path: str = "third_party/vjepa2",
        eps: float = 1e-5,
    ):
        super().__init__()
        encoder_type = str(encoder_type).strip().lower()
        if encoder_type != _VJEPA_ENCODER_TYPE:
            raise ValueError(
                f"Unsupported V-JEPA encoder_type={encoder_type!r}; "
                f"expected {_VJEPA_ENCODER_TYPE!r}"
            )
        encoder_file = _require_file(encoder_path, "V-JEPA 2.1 checkpoint")
        repo_dir = _require_dir(vjepa2_repo_path, "V-JEPA 2 source")
        stats_file = _require_file(stats_path, "V-JEPA latent stats")
        stats = torch.load(stats_file, map_location="cpu", weights_only=False)
        if not isinstance(stats, dict) or set(stats) < {"mean", "var"}:
            raise ValueError(
                f"V-JEPA stats must contain tensor mean and var: {stats_file}"
            )
        mean, var = stats["mean"], stats["var"]
        expected_shape = (VJEPA21ViTGEncoder.hidden_size, 24, 20)
        if (
            not torch.is_tensor(mean)
            or not torch.is_tensor(var)
            or tuple(mean.shape) != expected_shape
            or tuple(var.shape) != expected_shape
        ):
            raise ValueError(
                f"Invalid V-JEPA stats in {stats_file}: mean={getattr(mean, 'shape', None)}, "
                f"var={getattr(var, 'shape', None)}, expected {expected_shape}"
            )
        metadata_errors = []
        for key, expected_value in (
            ("dataset", str(stats_dataset)),
            ("representation", encoder_type),
            ("grid_size", (24, 20)),
            ("input_size", (384, 320)),
        ):
            actual = stats.get(key)
            if key in {"grid_size", "input_size"}:
                actual = tuple(actual or ())
            if actual != expected_value:
                metadata_errors.append(
                    f"{key}={stats.get(key)!r} (expected {expected_value!r})"
                )
        if not (
            stats.get("complete_train_split") is True
            or stats.get("formal_eligible") is True
        ):
            metadata_errors.append("stats are not marked formal/complete")
        if not torch.isfinite(mean).all() or not torch.isfinite(var).all():
            metadata_errors.append("mean/var contain non-finite values")
        if (var < 0).any():
            metadata_errors.append("variance contains negative values")
        if metadata_errors:
            raise ValueError(
                f"Invalid V-JEPA stats metadata in {stats_file}: "
                + "; ".join(metadata_errors)
            )

        self.encoder_type = encoder_type
        self.encoder = VJEPA21ViTGEncoder(
            checkpoint_path=encoder_file,
            repo_path=repo_dir,
        )
        self.register_buffer("latent_mean", mean.float().unsqueeze(0), persistent=False)
        self.register_buffer("latent_var", var.float().unsqueeze(0), persistent=False)
        self.eps = float(eps)
        self.model = SimpleNamespace(z_dim=VJEPA21ViTGEncoder.hidden_size)
        self.checkpoint_paths = {
            "encoder": str(encoder_file),
            "stats": str(stats_file),
            "source": str(repo_dir),
        }
        self.eval()
        self.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    def _stats_for_grid(
        self, grid: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(self.latent_mean.shape[-2:]) != tuple(grid):
            raise ValueError(
                f"V-JEPA stats grid {tuple(self.latent_mean.shape[-2:])} does not "
                f"match encoded grid {grid}; recompute stats at the exact resolution"
            )
        return (
            self.latent_mean.to(device=device, dtype=dtype),
            self.latent_var.to(device=device, dtype=dtype),
        )

    @torch.no_grad()
    def encode(self, video: torch.Tensor | list[torch.Tensor], **_: Any):
        return_list = isinstance(video, list)
        if return_list:
            if not video:
                raise ValueError("V-JEPA encode received an empty video list")
            video = torch.stack(video, dim=0)
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(
                f"V-JEPA encode expects [B,3,T,H,W], got {tuple(video.shape)}"
            )
        batch, _, frames, height, width = video.shape
        if height % 16 or width % 16:
            raise ValueError(
                f"V-JEPA input must be divisible by 16, got {height}x{width}"
            )
        encoder_input = build_causal_pairs(video).permute(0, 2, 1, 3, 4, 5).reshape(
            batch * frames, 3, 2, height, width
        )
        tokens = self.encoder(encoder_input)
        grid = (height // 16, width // 16)
        expected = (batch * frames, math.prod(grid), self.model.z_dim)
        if tuple(tokens.shape) != expected:
            raise RuntimeError(
                f"{self.encoder_type} produced {tuple(tokens.shape)}, expected {expected}"
            )
        latents = tokens.transpose(1, 2).reshape(
            batch, frames, self.model.z_dim, *grid
        ).permute(0, 2, 1, 3, 4).contiguous()
        mean, var = self._stats_for_grid(grid, latents.dtype, latents.device)
        latents = (latents - mean.unsqueeze(2)) / torch.sqrt(
            var.unsqueeze(2) + self.eps
        )
        # The official final normalization returns fp32 even under BF16
        # autocast. FastWAM's Video DiT boundary is BF16, including inference
        # paths that may not have an outer autocast context, so return the
        # normalized latent in the caller's visual dtype.
        latents = latents.to(dtype=video.dtype)
        if return_list:
            return [item for item in latents]
        return latents

    def decode(self, latents: torch.Tensor, **_: Any) -> torch.Tensor:
        raise NotImplementedError(
            "V-JEPA 2.1 has no RGB decoder. Training, joint/IDM latent loss, "
            "and RoboTwin action evaluation do not call decode(). Use "
            "decode_pca() only for diagnostic pseudo-colour visualization."
        )

    @torch.no_grad()
    def decode_pca(
        self,
        latents: torch.Tensor,
        output_size: tuple[int, int] = (384, 320),
        seed: int = 0,
    ) -> torch.Tensor:
        """Map normalized patch latents to deterministic PCA pseudo-colour RGB.

        The return value is [B,3,T,H,W] in [-1,1].  It is not a reconstruction
        and must never be used as a training target or quantitative prediction.
        """
        if latents.ndim != 5 or latents.shape[1] != self.model.z_dim:
            raise ValueError(
                f"PCA visualization expects [B,{self.model.z_dim},T,H,W], got "
                f"{tuple(latents.shape)}"
            )
        return pca_visualize_latents(latents, output_size=output_size, seed=seed)


def create_vjepa21_video_tokenizer(
    config: dict[str, Any],
    device: str | torch.device,
    torch_dtype: torch.dtype,
) -> VJEPA21VideoTokenizer:
    tokenizer = VJEPA21VideoTokenizer(**config)
    tokenizer.to(device=device)
    tokenizer.encoder.to(device=device, dtype=torch_dtype)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer

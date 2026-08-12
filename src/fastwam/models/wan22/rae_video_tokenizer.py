"""Frame-wise RAE/RAEv2 visual tokenizer for FastWAM.

The wrapper intentionally exposes the small subset of the Wan VAE interface
used by FastWAM.  Images stay rectangular: a [B,3,T,H,W] video is encoded as
B*T independent frames and decoded with the original H/16 by W/16 token grid.
"""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import SiglipVisionModel, ViTMAEForPreTraining
from transformers.models.vit_mae.configuration_vit_mae import ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import ViTMAELayer

from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_SIGLIP_MEAN = (0.5, 0.5, 0.5)
_SIGLIP_STD = (0.5, 0.5, 0.5)


_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve_candidates(path: str | Path) -> list[Path]:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return [raw.resolve()]
    return [raw.resolve(), (_PROJECT_ROOT / raw).resolve()]


def _require_file(path: str | Path, label: str) -> Path:
    candidates = _resolve_candidates(path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{label} not found: " + " | ".join(str(c) for c in candidates)
    )


def _require_dir(path: str | Path, label: str) -> Path:
    candidates = _resolve_candidates(path)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"{label} directory not found: " + " | ".join(str(c) for c in candidates)
    )


def _image_normalize(
    images: torch.Tensor,
    mean: Iterable[float],
    std: Iterable[float],
    dtype: torch.dtype,
) -> torch.Tensor:
    # FastWAM datasets use [-1, 1]; official RAE encoders receive [0, 1]
    # before their encoder-specific channel normalization.
    images = (images.to(dtype=dtype) + 1.0) * 0.5
    mean_t = images.new_tensor(tuple(mean)).view(1, 3, 1, 1)
    std_t = images.new_tensor(tuple(std)).view(1, 3, 1, 1)
    return (images - mean_t) / std_t


class _DINOv3K7Encoder(nn.Module):
    """Official DINOv3-L/16 with K7 patch-token elementwise summation."""

    layers = (11, 13, 15, 17, 19, 21, 23)
    hidden_size = 1024
    patch_size = 16

    def __init__(self, checkpoint_path: Path, repo_path: Path):
        super().__init__()
        hubconf = repo_path / "hubconf.py"
        if not hubconf.is_file():
            raise FileNotFoundError(
                "DINOv3 source checkout is invalid: expected hubconf.py at "
                f"{hubconf}. Set `model.vision_tokenizer_config.dinov3_repo_path` "
                "or DINOV3_REPO_PATH to the official facebookresearch/dinov3 checkout."
            )
        # Import the official backbone module directly. The repository hubconf
        # eagerly imports optional segmentation dependencies (torchmetrics,
        # mmcv, ...), which are unrelated to the encoder and need not be
        # installed for FastWAM.
        repo_str = str(repo_path)
        inserted = repo_str not in sys.path
        if inserted:
            sys.path.insert(0, repo_str)
        try:
            from dinov3.hub.backbones import dinov3_vitl16

            self.model = dinov3_vitl16(pretrained=False)
        finally:
            if inserted:
                sys.path.remove(repo_str)
        state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        self.model.load_state_dict(state, strict=True)
        # RAEv2's DINOv3 recipe removes affine parameters from the final norm.
        self.model.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = _image_normalize(images, _IMAGENET_MEAN, _IMAGENET_STD, images.dtype)
        outputs = self.model.get_intermediate_layers(
            images,
            n=list(self.layers),
            reshape=False,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        if len(outputs) != len(self.layers):
            raise RuntimeError(
                f"DINOv3 K7 expected {len(self.layers)} layer outputs, got {len(outputs)}"
            )
        # get_intermediate_layers strips CLS and all storage/register tokens.
        return torch.stack(outputs, dim=0).sum(dim=0)


class _SigLIP2Encoder(nn.Module):
    hidden_size = 768
    patch_size = 16

    def __init__(self, checkpoint_dir: Path):
        super().__init__()
        self.model = SiglipVisionModel.from_pretrained(
            str(checkpoint_dir), local_files_only=True
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = _image_normalize(images, _SIGLIP_MEAN, _SIGLIP_STD, images.dtype)
        output = self.model(pixel_values=images, interpolate_pos_encoding=True)
        # SigLIP2 vision embeddings contain spatial patches only (no CLS/register token).
        return output.last_hidden_state


class _MAEEncoder(nn.Module):
    hidden_size = 768
    patch_size = 16

    def __init__(self, checkpoint_dir: Path):
        super().__init__()
        self.model = ViTMAEForPreTraining.from_pretrained(
            str(checkpoint_dir), local_files_only=True
        ).vit
        self.model.layernorm.elementwise_affine = False
        self.model.layernorm.weight = None
        self.model.layernorm.bias = None
        self.model.config.mask_ratio = 0.0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = _image_normalize(images, _IMAGENET_MEAN, _IMAGENET_STD, images.dtype)
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"MAE input must be divisible by {self.patch_size}, got {height}x{width}"
            )
        patch_count = (height // self.patch_size) * (width // self.patch_size)
        # mask_ratio=0 plus deterministic ordered noise guarantees no random masking.
        noise = torch.arange(patch_count, device=images.device, dtype=images.dtype)
        noise = noise.unsqueeze(0).expand(images.shape[0], -1)
        output = self.model(images, noise=noise, interpolate_pos_encoding=True)
        return output.last_hidden_state[:, 1:]  # remove CLS


class RectangularRAEDecoder(nn.Module):
    """RAE ViTXL decoder with a dynamic rectangular patch grid."""

    def __init__(self, latent_dim: int, base_grid_size: tuple[int, int] = (16, 16)):
        super().__init__()
        self.base_grid_size = tuple(int(v) for v in base_grid_size)
        num_patches = math.prod(self.base_grid_size)
        config = ViTMAEConfig(
            hidden_size=int(latent_dim),
            decoder_hidden_size=1152,
            decoder_intermediate_size=4096,
            decoder_num_attention_heads=16,
            decoder_num_hidden_layers=28,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            layer_norm_eps=1e-12,
            qkv_bias=True,
            patch_size=16,
            num_channels=3,
        )
        self.config = config
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size),
            requires_grad=False,
        )
        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            ViTMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)
        )
        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size,
            config.patch_size**2 * config.num_channels,
        )
        self.trainable_cls_token = nn.Parameter(
            torch.zeros(1, 1, config.decoder_hidden_size)
        )

    def _position_embedding(self, grid_size: tuple[int, int]) -> torch.Tensor:
        grid_h, grid_w = grid_size
        base_h, base_w = self.base_grid_size
        if (grid_h, grid_w) == (base_h, base_w):
            return self.decoder_pos_embed
        cls_pos = self.decoder_pos_embed[:, :1]
        patch_pos = self.decoder_pos_embed[:, 1:].reshape(
            1, base_h, base_w, self.decoder_pos_embed.shape[-1]
        )
        patch_pos = patch_pos.permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=(grid_h, grid_w),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(
            1, grid_h * grid_w, self.decoder_pos_embed.shape[-1]
        )
        return torch.cat((cls_pos, patch_pos), dim=1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4:
            raise ValueError(f"RAE decoder expects [B,C,H,W], got {tuple(latents.shape)}")
        batch, channels, grid_h, grid_w = latents.shape
        if channels != self.config.hidden_size:
            raise ValueError(
                f"RAE decoder channel mismatch: expected {self.config.hidden_size}, got {channels}"
            )
        tokens = latents.flatten(2).transpose(1, 2)
        tokens = self.decoder_embed(tokens)
        cls = self.trainable_cls_token.expand(batch, -1, -1)
        hidden_states = torch.cat((cls, tokens), dim=1)
        hidden_states = hidden_states + self._position_embedding((grid_h, grid_w)).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        for layer in self.decoder_layers:
            hidden_states = layer(
                hidden_states, head_mask=None, output_attentions=False
            )[0]
        logits = self.decoder_pred(self.decoder_norm(hidden_states))[:, 1:]
        return self.unpatchify(logits, (grid_h, grid_w))

    def unpatchify(
        self, patchified_pixels: torch.Tensor, grid_size: tuple[int, int]
    ) -> torch.Tensor:
        grid_h, grid_w = grid_size
        patch = int(self.config.patch_size)
        channels = int(self.config.num_channels)
        expected = grid_h * grid_w
        if patchified_pixels.shape[1] != expected:
            raise ValueError(
                f"Decoder token count mismatch: got {patchified_pixels.shape[1]}, "
                f"expected {grid_h}*{grid_w}={expected}"
            )
        pixels = patchified_pixels.reshape(
            patchified_pixels.shape[0], grid_h, grid_w, patch, patch, channels
        )
        pixels = torch.einsum("nhwpqc->nchpwq", pixels)
        return pixels.reshape(
            patchified_pixels.shape[0], channels, grid_h * patch, grid_w * patch
        )


def _load_decoder(
    decoder_path: Path,
    latent_dim: int,
    allow_mask_token: bool,
) -> RectangularRAEDecoder:
    decoder = RectangularRAEDecoder(latent_dim=latent_dim)
    state = torch.load(decoder_path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(state, dict):
        raise TypeError(
            f"RAE decoder checkpoint must be a state dict, got {type(state)}: {decoder_path}"
        )
    state = dict(state)
    ignored = []
    if "mask_token" in state and allow_mask_token:
        ignored.append("mask_token")
        state.pop("mask_token")
    expected = decoder.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = {
        key: (tuple(state[key].shape), tuple(expected[key].shape))
        for key in set(state) & set(expected)
        if tuple(state[key].shape) != tuple(expected[key].shape)
    }
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"Incompatible RAE decoder checkpoint {decoder_path}: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
        )
    decoder.load_state_dict(state, strict=True)
    if ignored:
        logger.info("Ignored legacy decoder-only keys in %s: %s", decoder_path, ignored)
    return decoder


class FramewiseRAEVideoTokenizer(nn.Module):
    """Wan-VAE-compatible wrapper around a frozen frame-wise RAE."""

    temporal_downsample_factor = 1
    upsampling_factor = 16

    def __init__(
        self,
        encoder_type: str,
        encoder_path: str,
        decoder_path: str,
        stats_path: str,
        stats_dataset: str = "robotwin_train",
        dinov3_repo_path: str | None = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        encoder_type = str(encoder_type).strip().lower()
        decoder_path_obj = _require_file(decoder_path, "RAE decoder checkpoint")
        stats_path_obj = _require_file(stats_path, "RAE latent stats")

        if encoder_type == "dinov3_k7":
            encoder_file = _require_file(encoder_path, "DINOv3 encoder checkpoint")
            if not dinov3_repo_path:
                raise ValueError(
                    "`dinov3_repo_path` is required for DINOv3. Point it to an "
                    "official facebookresearch/dinov3 source checkout."
                )
            repo_dir = _require_dir(dinov3_repo_path, "DINOv3 source")
            latent_dim = 1024
        elif encoder_type == "siglip2":
            encoder_dir = _require_dir(encoder_path, "SigLIP2 encoder")
            _require_file(encoder_dir / "model.safetensors", "SigLIP2 weights")
            latent_dim = 768
        elif encoder_type == "mae":
            encoder_dir = _require_dir(encoder_path, "MAE encoder")
            if not (encoder_dir / "model.safetensors").is_file() and not (
                encoder_dir / "pytorch_model.bin"
            ).is_file():
                raise FileNotFoundError(
                    f"MAE weights not found under {encoder_dir}; expected "
                    "model.safetensors or pytorch_model.bin"
                )
            latent_dim = 768
        else:
            raise ValueError(
                f"Unsupported RAE encoder_type={encoder_type!r}; "
                "expected one of: dinov3_k7, siglip2, mae"
            )

        stats = torch.load(stats_path_obj, map_location="cpu", weights_only=False)
        if not isinstance(stats, dict) or set(stats) < {"mean", "var"}:
            raise ValueError(
                f"RAE stats must be a dict containing mean and var: {stats_path_obj}"
            )
        mean, var = stats["mean"], stats["var"]
        if not torch.is_tensor(mean) or not torch.is_tensor(var):
            raise TypeError(f"RAE stats mean/var must be tensors: {stats_path_obj}")
        if mean.shape != var.shape or mean.ndim != 3 or mean.shape[0] != latent_dim:
            raise ValueError(
                f"Invalid RAE stats shapes in {stats_path_obj}: "
                f"mean={tuple(mean.shape)}, var={tuple(var.shape)}, expected [C,H,W] "
                f"with C={latent_dim}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(var).all():
            raise ValueError(f"RAE stats contain non-finite values: {stats_path_obj}")
        if (var < 0).any():
            raise ValueError(f"RAE stats variance contains negative values: {stats_path_obj}")
        expected_representation = {
            "dinov3_k7": "dinov3_k7",
            "siglip2": "siglip2_b",
            "mae": "mae_b",
        }[encoder_type]
        metadata_errors = []
        expected_stats_dataset = str(stats_dataset)
        if stats.get("dataset") != expected_stats_dataset:
            metadata_errors.append(
                f"dataset={stats.get('dataset')!r} "
                f"(expected {expected_stats_dataset!r})"
            )
        if stats.get("representation") != expected_representation:
            metadata_errors.append(
                f"representation={stats.get('representation')!r} "
                f"(expected {expected_representation!r})"
            )
        if tuple(stats.get("grid_size", ())) != tuple(mean.shape[-2:]):
            metadata_errors.append(
                f"grid_size={stats.get('grid_size')!r} "
                f"(tensor grid is {tuple(mean.shape[-2:])})"
            )
        if tuple(stats.get("input_size", ())) != (384, 320):
            metadata_errors.append(
                f"input_size={stats.get('input_size')!r} (expected (384, 320))"
            )
        if not (
            stats.get("complete_train_split") is True
            or stats.get("formal_eligible") is True
        ):
            metadata_errors.append(
                "neither complete_train_split nor formal_eligible is true "
                "(debug/insufficient sampled stats cannot be used for formal training)"
            )
        if metadata_errors:
            raise ValueError(
                f"Invalid formal RoboTwin stats metadata in {stats_path_obj}: "
                + "; ".join(metadata_errors)
                + ". Generate this file with scripts/compute_robotwin_rae_stats.py."
            )

        # Validate every path and the small stats file before allocating the
        # large frozen encoder/decoder networks.
        if encoder_type == "dinov3_k7":
            encoder = _DINOv3K7Encoder(encoder_file, repo_dir)
        elif encoder_type == "siglip2":
            encoder = _SigLIP2Encoder(encoder_dir)
        else:
            encoder = _MAEEncoder(encoder_dir)
        decoder = _load_decoder(
            decoder_path_obj,
            latent_dim=latent_dim,
            allow_mask_token=(encoder_type == "mae"),
        )

        self.encoder_type = encoder_type
        self.encoder = encoder
        self.decoder = decoder
        self.register_buffer("latent_mean", mean.float().unsqueeze(0), persistent=False)
        self.register_buffer("latent_var", var.float().unsqueeze(0), persistent=False)
        self.eps = float(eps)
        self.model = SimpleNamespace(z_dim=latent_dim)
        self.checkpoint_paths = {
            "encoder": str(Path(encoder_path).expanduser().resolve()),
            "decoder": str(decoder_path_obj),
            "stats": str(stats_path_obj),
        }
        self.eval()
        self.requires_grad_(False)

    def train(self, mode: bool = True):
        # The vision encoder and decoder are always frozen/eval, even when the
        # parent FastWAM module switches modes.
        super().train(False)
        self.encoder.eval()
        self.decoder.eval()
        return self

    def _stats_for_grid(
        self, grid_size: tuple[int, int], dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.latent_mean
        var = self.latent_var
        if mean.shape[-2:] != grid_size:
            raise ValueError(
                "RoboTwin training stats grid does not match the encoded latent "
                f"grid: stats={tuple(mean.shape[-2:])}, latent={grid_size}. "
                "Recompute stats at the exact training mosaic resolution; "
                "position-wise statistics are never interpolated."
            )
        return (
            mean.to(device=device, dtype=dtype),
            var.to(device=device, dtype=dtype),
        )

    @torch.no_grad()
    def encode(self, video: torch.Tensor | list[torch.Tensor], **_: Any):
        return_list = isinstance(video, list)
        if return_list:
            if not video:
                raise ValueError("RAE encode received an empty video list")
            video = torch.stack(video, dim=0)
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(
                f"RAE encode expects [B,3,T,H,W], got {tuple(video.shape)}"
            )
        batch, _, frames, height, width = video.shape
        if height % 16 or width % 16:
            raise ValueError(f"RAE input must be divisible by 16, got {height}x{width}")
        images = video.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, 3, height, width
        )
        tokens = self.encoder(images)
        grid = (height // 16, width // 16)
        expected_tokens = math.prod(grid)
        if tokens.shape != (batch * frames, expected_tokens, self.model.z_dim):
            raise RuntimeError(
                f"{self.encoder_type} produced {tuple(tokens.shape)}, expected "
                f"({batch * frames}, {expected_tokens}, {self.model.z_dim}) for grid {grid}"
            )
        latents = tokens.transpose(1, 2).reshape(
            batch, frames, self.model.z_dim, *grid
        ).permute(0, 2, 1, 3, 4).contiguous()
        mean, var = self._stats_for_grid(grid, latents.dtype, latents.device)
        latents = (latents - mean.unsqueeze(2)) / torch.sqrt(
            var.unsqueeze(2) + self.eps
        )
        if return_list:
            return [item for item in latents]
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, **_: Any) -> torch.Tensor:
        if latents.ndim != 5 or latents.shape[1] != self.model.z_dim:
            raise ValueError(
                f"RAE decode expects [B,{self.model.z_dim},T,H,W], got "
                f"{tuple(latents.shape)}"
            )
        batch, channels, frames, grid_h, grid_w = latents.shape
        mean, var = self._stats_for_grid(
            (grid_h, grid_w), latents.dtype, latents.device
        )
        latents = latents * torch.sqrt(var.unsqueeze(2) + self.eps) + mean.unsqueeze(2)
        frame_latents = latents.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, grid_h, grid_w
        )
        images = self.decoder(frame_latents)
        mean_rgb = images.new_tensor(
            _SIGLIP_MEAN if self.encoder_type == "siglip2" else _IMAGENET_MEAN
        ).view(1, 3, 1, 1)
        std_rgb = images.new_tensor(
            _SIGLIP_STD if self.encoder_type == "siglip2" else _IMAGENET_STD
        ).view(1, 3, 1, 1)
        images = images * std_rgb + mean_rgb
        images = images * 2.0 - 1.0
        return images.reshape(
            batch, frames, 3, grid_h * 16, grid_w * 16
        ).permute(0, 2, 1, 3, 4).contiguous()


def create_framewise_rae_video_tokenizer(
    config: dict[str, Any],
    device: str | torch.device,
    torch_dtype: torch.dtype,
) -> FramewiseRAEVideoTokenizer:
    tokenizer = FramewiseRAEVideoTokenizer(**config)
    # Keep statistics in fp32; cast only the frozen networks.
    tokenizer.to(device=device)
    tokenizer.encoder.to(device=device, dtype=torch_dtype)
    tokenizer.decoder.to(device=device, dtype=torch_dtype)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer

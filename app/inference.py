from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model_registry import CLASS_NAMES, ModelSpec
from src.models import ResNetUNet, TransUNet
from src.utils import _normalize_image, count_parameters, load_checkpoint


IMAGE_KEYS = ("image", "images", "img", "imgs", "x", "X", "ct", "volume")
NUM_CLASSES = len(CLASS_NAMES)


@dataclass
class LoadedModel:
    model: nn.Module
    spec: ModelSpec
    checkpoint_path: Path
    source: str
    trainable_parameters: int


def get_inference_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_model_info(model_dir: Path) -> dict[str, Any]:
    candidates = [model_dir / "model_info.json", *sorted(model_dir.rglob("model_info.json"))]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def find_checkpoint(model_dir: Path, checkpoint_name: str) -> Path:
    preferred = sorted(model_dir.rglob(checkpoint_name))
    if preferred:
        return preferred[0]

    candidates = sorted(model_dir.rglob("*.pt")) + sorted(model_dir.rglob("*.pth"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No .pt/.pth checkpoint found under {model_dir}")


def resolve_model_dir(spec: ModelSpec) -> tuple[Path, str]:
    if spec.local_dir:
        model_dir = Path(spec.local_dir).expanduser()
        if not model_dir.exists():
            raise FileNotFoundError(
                f"{spec.local_dir_env} points to a missing folder: {model_dir}"
            )
        return model_dir, "local"

    try:
        import kagglehub
        from kagglehub.exceptions import DataCorruptionError
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to download model checkpoints. "
            "Install requirements.txt or set a local *_MODEL_DIR environment variable."
        ) from exc

    try:
        return Path(kagglehub.model_download(spec.handle)), "kagglehub"
    except DataCorruptionError:
        return Path(kagglehub.model_download(spec.handle, force_download=True)), "kagglehub"


def build_model(spec: ModelSpec, model_info: dict[str, Any] | None = None) -> nn.Module:
    model_info = model_info or {}
    architecture = model_info.get("architecture", spec.architecture)
    num_classes = int(model_info.get("num_classes", NUM_CLASSES))
    in_channels = int(model_info.get("in_channels", 3))
    decoder_dropout = float(model_info.get("decoder_dropout", 0.1))

    if architecture == "ResNetUNet":
        return ResNetUNet(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained_encoder=False,
            decoder_dropout=decoder_dropout,
        )
    if architecture == "TransUNet":
        return TransUNet(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained_encoder=False,
            pretrained_transformer=False,
            transformer_model_name=model_info.get(
                "transformer_model_name", spec.transformer_model_name
            ),
            freeze_first_n_transformer_blocks=int(
                model_info.get("freeze_first_n_transformer_blocks", 0)
            ),
            decoder_dropout=decoder_dropout,
        )
    raise ValueError(f"Unsupported model architecture: {architecture}")


def load_segmentation_model(spec: ModelSpec, device: torch.device) -> LoadedModel:
    model_dir, source = resolve_model_dir(spec)
    model_info = read_model_info(model_dir)
    checkpoint_path = find_checkpoint(model_dir, spec.checkpoint_name)
    model = build_model(spec, model_info=model_info).to(device)
    load_checkpoint(checkpoint_path, model=model, map_location=device)
    model.eval()
    return LoadedModel(
        model=model,
        spec=spec,
        checkpoint_path=checkpoint_path,
        source=source,
        trainable_parameters=count_parameters(model),
    )


def preprocess_slice(
    image: np.ndarray,
    image_size: tuple[int, int] = (224, 224),
    hu_window: tuple[float, float] | None = (-125.0, 275.0),
) -> tuple[torch.Tensor, np.ndarray]:
    tensor = _normalize_image(image, hu_window=hu_window)
    if tuple(tensor.shape[-2:]) != image_size:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    display = tensor.detach().cpu()
    if display.shape[0] >= 3 and torch.allclose(display[0], display[1]):
        display_np = display[0].numpy()
    elif display.shape[0] >= 3:
        display_np = display[:3].permute(1, 2, 0).numpy()
    else:
        display_np = display[0].numpy()
    return tensor, np.clip(display_np, 0.0, 1.0)


@torch.no_grad()
def predict_mask(
    loaded_model: LoadedModel,
    image: np.ndarray,
    device: torch.device,
    hu_window: tuple[float, float] | None = (-125.0, 275.0),
) -> tuple[np.ndarray, np.ndarray]:
    tensor, display_image = preprocess_slice(image, hu_window=hu_window)
    logits = loaded_model.model(tensor.unsqueeze(0).to(device))
    logits = F.interpolate(
        logits,
        size=tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    mask = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    return mask, display_image


def colorize_mask(mask: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    cmap = plt.get_cmap("tab20", NUM_CLASSES)
    rgba = cmap(mask / max(1, NUM_CLASSES - 1))
    rgba[..., 3] = np.where(mask == 0, 0.0, alpha)
    return rgba


def colorize_mask_rgb(mask: np.ndarray) -> np.ndarray:
    rgba = colorize_mask(mask, alpha=1.0)
    rgb = rgba[..., :3]
    rgb[mask == 0] = 0.0
    return rgb


def make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if image.ndim == 2:
        base = np.stack([image, image, image], axis=-1)
    else:
        base = image[..., :3]

    overlay = base.copy()
    rgba = colorize_mask(mask, alpha=alpha)
    colors = rgba[..., :3]
    weights = rgba[..., 3:4]
    overlay = overlay * (1.0 - weights) + colors * weights
    return np.clip(overlay, 0.0, 1.0)


def legend_rows() -> list[tuple[str, str]]:
    cmap = plt.get_cmap("tab20", NUM_CLASSES)
    rows: list[tuple[str, str]] = []
    for class_id, name in CLASS_NAMES.items():
        color = cmap(class_id / max(1, NUM_CLASSES - 1))[:3]
        rgb = tuple(int(round(channel * 255)) for channel in color)
        rows.append((name, f"rgb{rgb}"))
    return rows

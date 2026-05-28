from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available training device."""

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""

    params = model.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Freeze or unfreeze all parameters in a module."""

    for param in module.parameters():
        param.requires_grad = requires_grad


def resize_logits_to_target(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Resize logits to the spatial shape of a mask tensor."""

    target_size = targets.shape[-2:]
    if logits.shape[-2:] == target_size:
        return logits
    return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


def get_resnet50(pretrained: bool = True) -> nn.Module:
    """Load torchvision ResNet-50 with compatibility across torchvision versions."""

    try:
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        return resnet50(weights=weights)
    except ImportError as exc:
        raise ImportError(
            "ResNet-UNet and TransUNet require torchvision. "
            "On Kaggle, install it with: pip install torchvision"
        ) from exc
    except TypeError:
        from torchvision.models import resnet50

        return resnet50(pretrained=pretrained)


def adapt_first_conv(model: nn.Module, in_channels: int) -> None:
    """Adapt a torchvision ResNet first conv layer for non-RGB inputs."""

    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        if in_channels == 1:
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        elif in_channels > 3:
            new_conv.weight[:, :3].copy_(old_conv.weight)
            mean_extra = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:].copy_(mean_extra.repeat(1, in_channels - 3, 1, 1))
        else:
            new_conv.weight.copy_(old_conv.weight[:, :in_channels])

    model.conv1 = new_conv


def save_checkpoint(path: str | Path, **state: Any) -> None:
    """Save a training checkpoint, creating parent folders when needed."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore model and optimizer state."""

    checkpoint = torch.load(path, map_location=map_location)
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def upload_kaggle_model_artifact(
    local_model_dir: str | Path,
    owner: str = "tensura3607",
    model_slug: str = "abdominal-multi-organ-segmentation",
    framework: str = "pyTorch",
    variation: str = "default",
    version_notes: Optional[str] = None,
    license_name: str = "Apache 2.0",
) -> str:
    """Upload a local artifact directory as a new Kaggle Model version."""

    local_model_dir = Path(local_model_dir)
    if not local_model_dir.exists():
        raise FileNotFoundError(f"Model artifact directory does not exist: {local_model_dir}")

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to upload Kaggle Models. "
            "Install it in Kaggle with: pip install kagglehub"
        ) from exc

    handle = f"{owner}/{model_slug}/{framework}/{variation}"
    kwargs: dict[str, Any] = {"license_name": license_name}
    if version_notes:
        kwargs["version_notes"] = version_notes
    kagglehub.model_upload(handle, str(local_model_dir), **kwargs)
    return handle


IMAGE_KEYS = ("image", "images", "img", "imgs", "x", "X", "ct", "volume")
MASK_KEYS = ("mask", "masks", "label", "labels", "seg", "segs", "y", "Y")
SUPPORTED_ARRAY_EXTENSIONS = {".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _pick_key(keys: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    key_set = set(keys)
    for candidate in candidates:
        if candidate in key_set:
            return candidate
    return None


def _load_image_file(path: str | Path, key_candidates: Sequence[str]) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        with np.load(path) as data:
            key = _pick_key(data.files, key_candidates)
            if key is None:
                raise KeyError(f"No compatible key found in {path}. Keys: {data.files}")
            return data[key]

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to load PNG/JPG/TIFF files.") from exc

    return np.asarray(Image.open(path))


def _stacked_array_records(image_path: Path, mask_path: Path) -> list[dict[str, Any]]:
    images = np.load(image_path, mmap_mode="r")
    masks = np.load(mask_path, mmap_mode="r")
    if len(images) != len(masks):
        raise ValueError(
            f"Stacked image/mask arrays have different lengths: {image_path} vs {mask_path}"
        )
    return [
        {
            "type": "array_pair",
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "index": index,
        }
        for index in range(len(images))
    ]


def _case_id_from_slice_name(name: str) -> str:
    return name.split("_slice", maxsplit=1)[0]


def _split_train_npz_by_case(
    train_dir: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = sorted(train_dir.glob("*.npz"))
    by_case: dict[str, list[Path]] = {}
    for path in files:
        by_case.setdefault(_case_id_from_slice_name(path.stem), []).append(path)

    cases = sorted(by_case)
    rng = random.Random(seed)
    rng.shuffle(cases)
    val_case_count = max(1, int(round(len(cases) * val_ratio))) if len(cases) > 1 else 0
    val_cases = set(cases[:val_case_count])

    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    for case_id, paths in by_case.items():
        target = val_records if case_id in val_cases else train_records
        target.extend({"type": "npz_file", "path": str(path)} for path in paths)
    return train_records, val_records


def _h5_slice_records(test_dir: Path) -> list[dict[str, Any]]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to read Synapse test_vol_h5 volumes. "
            "On Kaggle it is usually preinstalled; otherwise run: pip install h5py"
        ) from exc

    records: list[dict[str, Any]] = []
    for path in sorted(test_dir.glob("*.h5")):
        with h5py.File(path, "r") as data:
            num_slices = int(data["image"].shape[0])
        records.extend(
            {"type": "h5_slice", "path": str(path), "index": index}
            for index in range(num_slices)
        )
    return records


def _normalize_image(image: np.ndarray, hu_window: Optional[tuple[float, float]]) -> torch.Tensor:
    image = np.asarray(image, dtype=np.float32)
    image = np.squeeze(image)

    if image.ndim == 2:
        image = image[None, :, :]
    elif image.ndim == 3 and image.shape[-1] <= 4:
        image = np.moveaxis(image, -1, 0)
    elif image.ndim == 3 and image.shape[0] <= 4:
        pass
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if hu_window is not None:
        low, high = hu_window
        image = np.clip(image, low, high)

    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    else:
        image = np.zeros_like(image, dtype=np.float32)

    tensor = torch.from_numpy(image).float()
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] > 3:
        tensor = tensor[:3]
    return tensor


def _normalize_mask(mask: np.ndarray) -> torch.Tensor:
    mask = np.asarray(mask)
    mask = np.squeeze(mask)

    if mask.ndim == 3:
        if mask.shape[0] <= 16:
            mask = np.argmax(mask, axis=0)
        elif mask.shape[-1] <= 16:
            mask = np.argmax(mask, axis=-1)
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")
    if mask.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")
    return torch.from_numpy(mask.astype(np.int64))


def _resize_sample(
    image: torch.Tensor,
    mask: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(image.shape[-2:]) != image_size:
        image = F.interpolate(
            image.unsqueeze(0),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    if tuple(mask.shape[-2:]) != image_size:
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=image_size,
            mode="nearest",
        ).squeeze(0).squeeze(0).long()
    return image, mask


def _apply_intensity_augmentation(image: torch.Tensor) -> torch.Tensor:
    scale = 0.9 + 0.2 * torch.rand(1).item()
    shift = -0.05 + 0.1 * torch.rand(1).item()
    noise = torch.randn_like(image) * 0.01
    return torch.clamp(image * scale + shift + noise, 0.0, 1.0)


class SegmentationSliceDataset(Dataset):
    """Dataset for processed 2D medical segmentation slices on Kaggle."""

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        image_size: tuple[int, int] = (224, 224),
        hu_window: Optional[tuple[float, float]] = (-125.0, 275.0),
        augment: bool = False,
    ) -> None:
        self.records = list(records)
        self.image_size = image_size
        self.hu_window = hu_window
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        if record["type"] == "pair":
            image = _load_image_file(record["image_path"], IMAGE_KEYS)
            mask = _load_image_file(record["mask_path"], MASK_KEYS)
        elif record["type"] == "array_pair":
            image = np.load(record["image_path"], mmap_mode="r")[record["index"]]
            mask = np.load(record["mask_path"], mmap_mode="r")[record["index"]]
        elif record["type"] == "npz_file":
            with np.load(record["path"]) as data:
                image = data["image"]
                mask = data["label"] if "label" in data.files else data["mask"]
        elif record["type"] == "h5_slice":
            try:
                import h5py
            except ImportError as exc:
                raise ImportError("h5py is required to read .h5 test volumes.") from exc
            with h5py.File(record["path"], "r") as data:
                image = data["image"][record["index"]]
                mask = data["label"][record["index"]]
        elif record["type"] == "npz":
            with np.load(record["path"]) as data:
                image = data[record["image_key"]]
                mask = data[record["mask_key"]]
                if record.get("index") is not None:
                    image = image[record["index"]]
                    mask = mask[record["index"]]
        else:
            raise ValueError(f"Unknown record type: {record['type']}")

        image_tensor = _normalize_image(image, self.hu_window)
        mask_tensor = _normalize_mask(mask)
        image_tensor, mask_tensor = _resize_sample(image_tensor, mask_tensor, self.image_size)
        if self.augment:
            image_tensor = _apply_intensity_augmentation(image_tensor)
        return image_tensor, mask_tensor


def _pair_records(image_dir: Path, mask_dir: Path) -> list[dict[str, Any]]:
    image_files = [
        path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_ARRAY_EXTENSIONS
    ]
    mask_files = [
        path for path in mask_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_ARRAY_EXTENSIONS
    ]
    masks_by_stem = {path.stem: path for path in mask_files}
    records = []
    for image_path in sorted(image_files):
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            continue
        records.append(
            {
                "type": "pair",
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            }
        )
    return records


def _npz_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with np.load(path) as data:
        image_key = _pick_key(data.files, IMAGE_KEYS)
        mask_key = _pick_key(data.files, MASK_KEYS)
        if image_key is None or mask_key is None:
            return records

        images = data[image_key]
        masks = data[mask_key]
        if images.ndim >= 3 and masks.ndim >= 3 and len(images) == len(masks):
            for index in range(len(images)):
                records.append(
                    {
                        "type": "npz",
                        "path": str(path),
                        "image_key": image_key,
                        "mask_key": mask_key,
                        "index": index,
                    }
                )
        else:
            records.append(
                {
                    "type": "npz",
                    "path": str(path),
                    "image_key": image_key,
                    "mask_key": mask_key,
                    "index": None,
                }
            )
    return records


def discover_segmentation_records(data_root: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Discover train/val/test records from common Kaggle segmentation layouts."""

    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")

    train_npz_dir = data_root / "train_npz"
    test_vol_dir = data_root / "test_vol_h5"
    if train_npz_dir.exists() and test_vol_dir.exists():
        train_records, val_records = _split_train_npz_by_case(train_npz_dir)
        test_records = _h5_slice_records(test_vol_dir)
        return {"train": train_records, "val": val_records, "test": test_records}

    split_records: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "valid", "validation", "test"):
        split_dir = data_root / split
        image_dir = split_dir / "images"
        mask_dir = split_dir / "masks"
        if image_dir.exists() and mask_dir.exists():
            name = "val" if split in {"valid", "validation"} else split
            split_records[name] = _pair_records(image_dir, mask_dir)

    if split_records:
        return split_records

    image_dir = data_root / "images"
    mask_dir = data_root / "masks"
    if image_dir.exists() and mask_dir.exists():
        return {"all": _pair_records(image_dir, mask_dir)}

    for image_name in ("images.npy", "x.npy", "X.npy"):
        for mask_name in ("masks.npy", "labels.npy", "y.npy", "Y.npy"):
            image_path = data_root / image_name
            mask_path = data_root / mask_name
            if image_path.exists() and mask_path.exists():
                return {"all": _stacked_array_records(image_path, mask_path)}

    records: list[dict[str, Any]] = []
    for path in sorted(data_root.rglob("*.npz")):
        records.extend(_npz_records(path))
    if records:
        return {"all": records}

    raise FileNotFoundError(
        "Could not find segmentation data. Expected images/masks folders, "
        "train|val|test/images and masks folders, or .npz files with image/mask keys."
    )


def split_records(
    split_records: dict[str, list[dict[str, Any]]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Return train/val/test records, random-splitting when only one pool exists."""

    if "train" in split_records and "val" in split_records and "test" in split_records:
        return {
            "train": split_records["train"],
            "val": split_records["val"],
            "test": split_records["test"],
        }

    if "train" in split_records and "test" in split_records:
        train_pool = list(split_records["train"])
        rng = random.Random(seed)
        rng.shuffle(train_pool)
        val_count = max(1, int(len(train_pool) * val_ratio)) if len(train_pool) > 1 else 0
        return {
            "train": train_pool[val_count:],
            "val": train_pool[:val_count],
            "test": split_records["test"],
        }

    records = list(split_records.get("all", []))
    if not records:
        for key in ("train", "val", "test"):
            records.extend(split_records.get(key, []))
    if not records:
        raise ValueError("No records found to split.")

    rng = random.Random(seed)
    rng.shuffle(records)
    total = len(records)
    train_count = max(1, int(total * train_ratio))
    val_count = max(1, int(total * val_ratio)) if total >= 3 else 0
    if train_count + val_count >= total:
        train_count = max(1, total - 2)
        val_count = 1 if total >= 2 else 0

    return {
        "train": records[:train_count],
        "val": records[train_count:train_count + val_count],
        "test": records[train_count + val_count:],
    }


def build_segmentation_dataloaders(
    data_root: str | Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 8,
    num_workers: int = 2,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    hu_window: Optional[tuple[float, float]] = (-125.0, 275.0),
) -> tuple[dict[str, DataLoader], dict[str, list[dict[str, Any]]]]:
    """Build train, validation, and test DataLoaders for Kaggle notebooks."""

    records = split_records(
        discover_segmentation_records(data_root),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    datasets = {
        "train": SegmentationSliceDataset(records["train"], image_size, hu_window, augment=True),
        "val": SegmentationSliceDataset(records["val"], image_size, hu_window, augment=False),
        "test": SegmentationSliceDataset(records["test"], image_size, hu_window, augment=False),
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }
    return loaders, records


def _empty_metric_state(num_classes: int) -> dict[str, torch.Tensor]:
    return {
        "tp": torch.zeros(num_classes, dtype=torch.float64),
        "fp": torch.zeros(num_classes, dtype=torch.float64),
        "fn": torch.zeros(num_classes, dtype=torch.float64),
        "correct": torch.zeros(1, dtype=torch.float64),
        "total": torch.zeros(1, dtype=torch.float64),
    }


def _update_metric_state(
    state: dict[str, torch.Tensor],
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> None:
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    state["correct"] += (preds == targets).sum().item()
    state["total"] += targets.numel()
    for class_id in range(num_classes):
        pred_mask = preds == class_id
        target_mask = targets == class_id
        state["tp"][class_id] += (pred_mask & target_mask).sum().item()
        state["fp"][class_id] += (pred_mask & ~target_mask).sum().item()
        state["fn"][class_id] += (~pred_mask & target_mask).sum().item()


def _finalize_metrics(
    state: dict[str, torch.Tensor],
    loss_total: float,
    num_batches: int,
    include_background: bool = False,
) -> dict[str, Any]:
    eps = 1e-7
    dice = (2 * state["tp"] + eps) / (2 * state["tp"] + state["fp"] + state["fn"] + eps)
    iou = (state["tp"] + eps) / (state["tp"] + state["fp"] + state["fn"] + eps)
    metric_slice = slice(None) if include_background else slice(1, None)
    return {
        "loss": loss_total / max(1, num_batches),
        "dice_mean": float(dice[metric_slice].mean().item()),
        "iou_mean": float(iou[metric_slice].mean().item()),
        "pixel_acc": float((state["correct"] / state["total"].clamp_min(1)).item()),
        "dice_per_class": [float(value) for value in dice.tolist()],
        "iou_per_class": [float(value) for value in iou.tolist()],
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    include_background_metrics: bool = False,
) -> dict[str, Any]:
    """Train for one epoch and return loss/Dice/IoU metrics."""

    model.train()
    state = _empty_metric_state(num_classes)
    loss_total = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                logits = resize_logits_to_target(logits, masks)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            logits = resize_logits_to_target(logits, masks)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        loss_total += float(loss.detach().cpu())
        _update_metric_state(state, logits.argmax(dim=1), masks, num_classes)

    return _finalize_metrics(state, loss_total, len(loader), include_background_metrics)


@torch.no_grad()
def evaluate_segmentation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    include_background_metrics: bool = False,
) -> dict[str, Any]:
    """Evaluate a segmentation model on validation or test data."""

    model.eval()
    state = _empty_metric_state(num_classes)
    loss_total = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        logits = resize_logits_to_target(logits, masks)
        loss = criterion(logits, masks)
        loss_total += float(loss.detach().cpu())
        _update_metric_state(state, logits.argmax(dim=1), masks, num_classes)

    return _finalize_metrics(state, loss_total, len(loader), include_background_metrics)


def fit_segmentation_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    epochs: int,
    checkpoint_path: str | Path,
    use_amp: bool = True,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 1e-4,
) -> list[dict[str, Any]]:
    """Run train/val loop, save the best validation Dice checkpoint, and optionally early-stop."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None
    history: list[dict[str, Any]] = []
    best_dice = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, num_classes, scaler=scaler
        )
        val_metrics = evaluate_segmentation(
            model, loaders["val"], criterion, device, num_classes
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice_mean']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice_mean']:.4f} "
            f"val_iou={val_metrics['iou_mean']:.4f}"
        )
        improved = val_metrics["dice_mean"] > best_dice + early_stopping_min_delta
        if improved:
            best_dice = val_metrics["dice_mean"]
            epochs_without_improvement = 0
            row["is_best"] = True
            save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                val_metrics=val_metrics,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            row["is_best"] = False
            if early_stopping_patience is not None:
                print(
                    "No validation Dice improvement "
                    f"({epochs_without_improvement}/{early_stopping_patience})."
                )

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                "Early stopping triggered: "
                f"best_val_dice={best_dice:.4f}, epoch={epoch}."
            )
            break

    return history


def _image_for_display(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu()
    if image.ndim == 3 and image.shape[0] == 1:
        return image[0].numpy()
    if image.ndim == 3 and image.shape[0] >= 3:
        channels = image[:3]
        if torch.allclose(channels[0], channels[1]) and torch.allclose(channels[1], channels[2]):
            return channels[0].numpy()
        return channels.permute(1, 2, 0).numpy()
    if image.ndim == 2:
        return image.numpy()
    raise ValueError(f"Unsupported display image shape: {tuple(image.shape)}")


def _mask_for_display(mask: torch.Tensor) -> np.ndarray:
    return mask.detach().cpu().numpy().astype(np.int64)


def _draw_image_mask_overlay(
    axes: Sequence[Any],
    image: torch.Tensor,
    mask: torch.Tensor,
    sample_title: str,
    num_classes: int,
) -> None:
    import matplotlib.pyplot as plt

    image_np = _image_for_display(image)
    mask_np = _mask_for_display(mask)
    colored_mask = np.ma.masked_where(mask_np == 0, mask_np)
    cmap = plt.get_cmap("tab20", num_classes)

    axes[0].imshow(image_np, cmap="gray" if image_np.ndim == 2 else None)
    axes[0].set_title(f"{sample_title} - image")

    axes[1].imshow(mask_np, cmap=cmap, vmin=0, vmax=num_classes - 1)
    axes[1].set_title(f"{sample_title} - mask")

    axes[2].imshow(image_np, cmap="gray" if image_np.ndim == 2 else None)
    axes[2].imshow(colored_mask, cmap=cmap, vmin=0, vmax=num_classes - 1, alpha=0.55)
    axes[2].set_title(f"{sample_title} - overlay")

    for axis in axes:
        axis.axis("off")


def show_loaded_segmentation_samples(
    loader: DataLoader,
    max_samples: int = 3,
    num_classes: int = 9,
    title: str = "Loaded samples",
    prefer_foreground: bool = True,
    max_batches_to_scan: int = 16,
) -> None:
    """Show original image, colored mask, and overlay for loaded dataset samples."""

    import matplotlib.pyplot as plt

    selected_images: list[torch.Tensor] = []
    selected_masks: list[torch.Tensor] = []

    for batch_index, (images, masks) in enumerate(loader):
        for index in range(images.shape[0]):
            has_foreground = bool(torch.any(masks[index] > 0).item())
            if not prefer_foreground or has_foreground:
                selected_images.append(images[index])
                selected_masks.append(masks[index])
            if len(selected_images) >= max_samples:
                break
        if len(selected_images) >= max_samples or batch_index + 1 >= max_batches_to_scan:
            break

    if not selected_images:
        images, masks = next(iter(loader))
        count = min(max_samples, images.shape[0])
        selected_images = [images[index] for index in range(count)]
        selected_masks = [masks[index] for index in range(count)]

    count = len(selected_images)
    print(f"{title} labels:", [torch.unique(selected_masks[index]).tolist() for index in range(count)])

    fig, axes = plt.subplots(count, 3, figsize=(12, 3.8 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle(title)

    for index in range(count):
        _draw_image_mask_overlay(
            axes[index],
            selected_images[index],
            selected_masks[index],
            sample_title=f"Sample {index + 1}",
            num_classes=num_classes,
        )

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def show_segmentation_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 4,
    num_classes: int = 9,
    title: str = "Test predictions",
    prefer_foreground: bool = True,
    max_batches_to_scan: int = 16,
) -> None:
    """Display image, ground-truth mask, and predicted mask for a few samples."""

    import matplotlib.pyplot as plt

    model.eval()
    selected_images: list[torch.Tensor] = []
    selected_masks: list[torch.Tensor] = []
    for batch_index, (batch_images, batch_masks) in enumerate(loader):
        for index in range(batch_images.shape[0]):
            has_foreground = bool(torch.any(batch_masks[index] > 0).item())
            if not prefer_foreground or has_foreground:
                selected_images.append(batch_images[index])
                selected_masks.append(batch_masks[index])
            if len(selected_images) >= max_samples:
                break
        if len(selected_images) >= max_samples or batch_index + 1 >= max_batches_to_scan:
            break

    if selected_images:
        images = torch.stack(selected_images, dim=0)
        masks = torch.stack(selected_masks, dim=0)
    else:
        images, masks = next(iter(loader))
        images = images[:max_samples]
        masks = masks[:max_samples]

    images = images.to(device)
    logits = model(images)
    logits = resize_logits_to_target(logits, masks.to(device))
    preds = logits.argmax(dim=1).cpu()
    images = images.cpu()

    count = min(max_samples, images.shape[0])
    cmap = plt.get_cmap("tab20", num_classes)
    fig, axes = plt.subplots(count, 4, figsize=(15, 3.8 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle(title)

    for index in range(count):
        image_np = _image_for_display(images[index])
        mask_np = _mask_for_display(masks[index])
        pred_np = _mask_for_display(preds[index])
        pred_overlay = np.ma.masked_where(pred_np == 0, pred_np)

        axes[index, 0].imshow(image_np, cmap="gray" if image_np.ndim == 2 else None)
        axes[index, 0].set_title(f"Sample {index + 1} - image")
        axes[index, 1].imshow(mask_np, cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[index, 1].set_title("Ground truth")
        axes[index, 2].imshow(pred_np, cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[index, 2].set_title("Prediction")
        axes[index, 3].imshow(image_np, cmap="gray" if image_np.ndim == 2 else None)
        axes[index, 3].imshow(pred_overlay, cmap=cmap, vmin=0, vmax=num_classes - 1, alpha=0.55)
        axes[index, 3].set_title("Prediction overlay")
        for axis in axes[index]:
            axis.axis("off")

    plt.tight_layout()
    plt.show()

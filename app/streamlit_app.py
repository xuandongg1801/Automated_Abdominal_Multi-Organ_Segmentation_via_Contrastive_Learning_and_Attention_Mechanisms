from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import (
    IMAGE_KEYS,
    colorize_mask_rgb,
    get_inference_device,
    legend_rows,
    load_segmentation_model,
    make_overlay,
    predict_mask,
)
from app.model_registry import CLASS_NAMES, DEFAULT_MODEL_LABEL, MODEL_SPECS


ARCHITECTURE_LABELS = {
    "ResNetUNet": "ResNet-UNet",
    "TransUNet": "TransUNet",
}


def hide_selectbox_text_cursor() -> None:
    st.markdown(
        """
        <style>
        div[data-baseweb="select"] input {
            caret-color: transparent !important;
            cursor: pointer !important;
        }
        div[data-baseweb="select"] [role="combobox"],
        div[data-baseweb="select"] svg {
            cursor: pointer !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_streamlit_secrets() -> None:
    try:
        secrets = st.secrets
        for key in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
            value = secrets.get(key)
            if value:
                os.environ.setdefault(key, str(value))
    except Exception as exc:
        if exc.__class__.__name__ != "StreamlitSecretNotFoundError":
            raise


def read_uploaded_arrays(uploaded_file) -> dict[str, np.ndarray]:
    suffix = Path(uploaded_file.name).suffix.lower()
    payload = uploaded_file.getvalue()

    if suffix == ".npy":
        return {"image": np.load(io.BytesIO(payload), allow_pickle=False)}
    if suffix == ".npz":
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    image = Image.open(io.BytesIO(payload))
    return {"image": np.asarray(image)}


def default_image_key(arrays: dict[str, np.ndarray]) -> str:
    for key in IMAGE_KEYS:
        if key in arrays:
            return key
    return next(iter(arrays))


def default_label_key(arrays: dict[str, np.ndarray], image_key: str) -> str | None:
    for key in ("label", "labels", "mask", "masks", "seg", "segs", "y", "Y"):
        if key in arrays and key != image_key:
            return key
    return None


def is_image_like(array: np.ndarray) -> bool:
    if array.ndim == 2:
        return True
    if array.ndim != 3:
        return False
    return array.shape[-1] in (3, 4) or array.shape[0] in (1, 3, 4)


def slice_count(array: np.ndarray) -> int:
    if is_image_like(array):
        return 1
    if array.ndim in (3, 4):
        return int(array.shape[0])
    raise ValueError(f"Unsupported array shape: {array.shape}")


def select_slice(array: np.ndarray, index: int) -> np.ndarray:
    if is_image_like(array):
        return array
    if array.ndim in (3, 4):
        return array[index]
    raise ValueError(f"Unsupported array shape: {array.shape}")


def class_id_mask_to_png_bytes(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8), mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def rgb_to_png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)).save(
        buffer,
        format="PNG",
    )
    return buffer.getvalue()


def image_to_png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)).save(
        buffer,
        format="PNG",
    )
    return buffer.getvalue()


def format_shape(array: np.ndarray) -> str:
    return " x ".join(str(dim) for dim in array.shape)


def format_range(array: np.ndarray) -> str:
    if array.size == 0:
        return "empty"
    if np.issubdtype(array.dtype, np.number):
        return f"{float(np.nanmin(array)):.3g} to {float(np.nanmax(array)):.3g}"
    return "n/a"


def format_class_list(class_ids: list[int]) -> str:
    if not class_ids:
        return "none"
    return ", ".join(
        f"{class_id} {CLASS_NAMES.get(class_id, f'Class {class_id}')}"
        for class_id in class_ids
    )


def summarize_mask(mask: np.ndarray) -> tuple[list[dict[str, object]], list[int], int]:
    values, counts = np.unique(mask.astype(np.int64), return_counts=True)
    rows: list[dict[str, object]] = []
    class_ids: list[int] = []
    foreground_pixels = 0
    for class_id, count in zip(values.tolist(), counts.tolist()):
        class_id = int(class_id)
        count = int(count)
        class_ids.append(class_id)
        if class_id != 0:
            foreground_pixels += count
        rows.append(
            {
                "Class ID": class_id,
                "Class": CLASS_NAMES.get(class_id, f"Class {class_id}"),
                "Pixels": count,
            }
        )
    return rows, class_ids, foreground_pixels


def normalize_label_mask(label: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(label)
    mask = np.squeeze(mask)

    if mask.ndim == 3:
        if mask.shape[0] <= 16:
            mask = np.argmax(mask, axis=0)
        elif mask.shape[-1] <= 16:
            mask = np.argmax(mask, axis=-1)
        else:
            raise ValueError(f"Unsupported label shape: {mask.shape}")

    if mask.ndim != 2:
        raise ValueError(f"Unsupported label shape: {mask.shape}")

    mask = np.clip(mask, 0, 255).astype(np.uint8)
    if tuple(mask.shape) != target_shape:
        image = Image.fromarray(mask, mode="L")
        image = image.resize((target_shape[1], target_shape[0]), Image.Resampling.NEAREST)
        mask = np.asarray(image, dtype=np.uint8)
    return mask


def segmentation_metric_rows(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> list[dict[str, object]]:
    class_ids = sorted(
        (set(np.unique(prediction).astype(int)) | set(np.unique(ground_truth).astype(int)))
        - {0}
    )
    rows: list[dict[str, object]] = []
    for class_id in class_ids:
        pred_class = prediction == class_id
        gt_class = ground_truth == class_id
        intersection = int(np.logical_and(pred_class, gt_class).sum())
        pred_pixels = int(pred_class.sum())
        gt_pixels = int(gt_class.sum())
        union = int(np.logical_or(pred_class, gt_class).sum())
        dice = 1.0 if pred_pixels + gt_pixels == 0 else (2.0 * intersection) / (
            pred_pixels + gt_pixels
        )
        iou = 1.0 if union == 0 else intersection / union
        rows.append(
            {
                "Class ID": class_id,
                "Class": CLASS_NAMES.get(class_id, f"Class {class_id}"),
                "GT pixels": gt_pixels,
                "Pred pixels": pred_pixels,
                "Dice": round(dice, 4),
                "IoU": round(iou, 4),
            }
        )
    return rows


def mean_metric(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows]
    if not values:
        return None
    return float(np.mean(values))


def architecture_options() -> list[str]:
    ordered = ["ResNetUNet", "TransUNet"]
    available = {spec.architecture for spec in MODEL_SPECS.values()}
    return [architecture for architecture in ordered if architecture in available]


def contrastive_options(architecture: str) -> list[float]:
    return sorted(
        {
            float(spec.contrastive_weight)
            for spec in MODEL_SPECS.values()
            if spec.architecture == architecture
        }
    )


def contrastive_label(weight: float) -> str:
    if weight == 0:
        return "Baseline (cw0)"
    return f"Contrastive cw={weight:g}"


def model_label_for(architecture: str, contrastive_weight: float) -> str:
    for label, spec in MODEL_SPECS.items():
        if (
            spec.architecture == architecture
            and float(spec.contrastive_weight) == float(contrastive_weight)
        ):
            return label
    raise KeyError(f"No model for {architecture} with cw={contrastive_weight}")


def render_inference_error(exc: Exception) -> None:
    name = exc.__class__.__name__
    message = str(exc)
    st.error(f"{name}: {message}")

    lower_message = message.lower()
    if name == "DataCorruptionError" or "checksum" in lower_message:
        st.info(
            "The checkpoint download looks corrupted. Clear the loaded model cache, "
            "then rerun inference so KaggleHub can retry the download."
        )
    elif "kaggle" in lower_message or "credential" in lower_message:
        st.info(
            "Check Kaggle credentials in Streamlit Secrets, or set the matching "
            "*_MODEL_DIR environment variable for a local checkpoint folder."
        )
    elif "cuda" in lower_message:
        st.info("CUDA failed in this environment. Switch Device to CPU and rerun.")
    else:
        st.info("Check the uploaded array shape and model checkpoint, then rerun inference.")


@st.cache_resource(show_spinner=False)
def cached_model(model_label: str, device_name: str):
    device = get_inference_device(prefer_cuda=(device_name == "cuda"))
    return load_segmentation_model(MODEL_SPECS[model_label], device), device


def main() -> None:
    st.set_page_config(
        page_title="Abdominal Multi-Organ Segmentation",
        layout="wide",
    )
    apply_streamlit_secrets()

    st.title("Abdominal Multi-Organ Segmentation")
    hide_selectbox_text_cursor()

    with st.sidebar:
        default_spec = MODEL_SPECS[DEFAULT_MODEL_LABEL]
        architectures = architecture_options()
        architecture = st.selectbox(
            "Architecture",
            architectures,
            index=architectures.index(default_spec.architecture),
            format_func=lambda value: ARCHITECTURE_LABELS.get(value, value),
            width="stretch",
        )
        weights = contrastive_options(architecture)
        default_weight = (
            float(default_spec.contrastive_weight)
            if architecture == default_spec.architecture
            else weights[0]
        )
        contrastive_weight = st.selectbox(
            "Training variant",
            weights,
            index=weights.index(default_weight) if default_weight in weights else 0,
            format_func=contrastive_label,
            width="stretch",
        )
        contrastive_weight = float(contrastive_weight)
        model_label = model_label_for(architecture, contrastive_weight)
        model_spec = MODEL_SPECS[model_label]

        cuda_available = torch.cuda.is_available()
        device_options = ["cpu", "cuda"] if cuda_available else ["cpu"]
        device_name = st.radio("Device", device_options, horizontal=True)
        if not cuda_available:
            st.caption("CUDA is unavailable here; inference will run on CPU.")

        use_hu_window = st.toggle(
            "Apply CT HU window [-125, 275]",
            value=True,
            help="Use this for raw CT values. Processed Synapse .npz slices are already normalized, so either setting should be stable.",
        )
        alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.5, 0.05)

        if st.button("Clear loaded model cache"):
            cached_model.clear()
            st.success("Loaded model cache cleared.")

        with st.expander("Model source"):
            st.caption("Kaggle handle")
            st.code(model_spec.handle, language=None)
            st.caption(f"Local override env: `{model_spec.local_dir_env}`")
            if model_spec.local_dir:
                st.caption(f"Using local checkpoint folder: `{model_spec.local_dir}`")

    uploaded_file = st.file_uploader(
        "Upload CT slice",
        type=("png", "jpg", "jpeg", "tif", "tiff", "npy", "npz"),
    )

    if uploaded_file is None:
        st.info("Upload a CT slice or processed Synapse array to run segmentation.")
        return

    try:
        arrays = read_uploaded_arrays(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read upload: {type(exc).__name__}: {exc}")
        st.stop()

    image_key = default_image_key(arrays)
    if len(arrays) > 1:
        image_key = st.selectbox(
            "Image array",
            list(arrays),
            index=list(arrays).index(image_key),
            format_func=lambda key: f"{key} | shape {format_shape(arrays[key])} | {arrays[key].dtype}",
        )

    array = arrays[image_key]
    try:
        count = slice_count(array)
        slice_index = 0
        if count > 1:
            slice_index = st.slider("Slice", 0, count - 1, min(count // 2, count - 1))
        image = select_slice(array, slice_index)
    except Exception as exc:
        st.error(f"Unsupported image array: {type(exc).__name__}: {exc}")
        st.stop()

    label_key = default_label_key(arrays, image_key)
    label_values: np.ndarray | None = None
    label_slice: np.ndarray | None = None
    if label_key is not None:
        try:
            label_slice = select_slice(arrays[label_key], slice_index)
            label_values = np.unique(label_slice.astype(np.int64))
        except Exception:
            label_values = None

    if label_values is not None:
        if np.all(label_values == 0):
            st.warning(
                f"`{label_key}` for this slice contains only background. "
                "A black predicted mask can be the correct result for this input."
            )
        else:
            st.caption(f"Ground-truth labels in `{label_key}`: {label_values.tolist()}")

    with st.expander("Input details", expanded=True):
        st.caption(f"File: `{uploaded_file.name}`")
        details = st.columns(4)
        details[0].metric("Array shape", format_shape(array))
        details[1].metric("Selected slice", f"{slice_index + 1}/{count}")
        details[2].metric("Image range", format_range(image))
        details[3].metric("Image dtype", str(array.dtype))
        st.caption(f"Image key: `{image_key}`")
        if label_key is not None:
            st.caption(f"Label key: `{label_key}` | label shape: {format_shape(arrays[label_key])}")

    hu_window = (-125.0, 275.0) if use_hu_window else None

    try:
        with st.spinner("Loading model checkpoint..."):
            loaded_model, device = cached_model(model_label, device_name)
        with st.spinner("Running inference..."):
            mask, display_image = predict_mask(
                loaded_model,
                image=image,
                device=device,
                hu_window=hu_window,
            )
    except Exception as exc:
        render_inference_error(exc)
        st.stop()

    ground_truth_mask: np.ndarray | None = None
    if label_slice is not None:
        try:
            ground_truth_mask = normalize_label_mask(label_slice, target_shape=mask.shape)
        except Exception as exc:
            st.warning(f"Could not render ground truth mask: {type(exc).__name__}: {exc}")

    overlay = make_overlay(display_image, mask, alpha=alpha)

    pred_rows, pred_class_ids, foreground_pixels = summarize_mask(mask)
    metric_rows = (
        segmentation_metric_rows(mask, ground_truth_mask)
        if ground_truth_mask is not None
        else []
    )
    mean_dice = mean_metric(metric_rows, "Dice")
    mean_iou = mean_metric(metric_rows, "IoU")

    if foreground_pixels == 0:
        st.warning(
            "Prediction contains only background. This can be correct for empty slices, "
            "but use a foreground slice when demonstrating model quality."
        )
    else:
        foreground_class_ids = [class_id for class_id in pred_class_ids if class_id != 0]
        st.caption(f"Predicted foreground labels: {format_class_list(foreground_class_ids)}")

    if ground_truth_mask is None:
        col_image, col_mask, col_overlay = st.columns(3)
        col_image.image(display_image, caption="Input", width="stretch", clamp=True)
        col_mask.image(
            colorize_mask_rgb(mask),
            caption="Predicted mask",
            width="stretch",
            clamp=True,
        )
        col_overlay.image(overlay, caption="Overlay", width="stretch", clamp=True)
    else:
        col_image, col_gt, col_mask, col_overlay = st.columns(4)
        col_image.image(display_image, caption="Input", width="stretch", clamp=True)
        col_gt.image(
            colorize_mask_rgb(ground_truth_mask),
            caption="Ground truth",
            width="stretch",
            clamp=True,
        )
        col_mask.image(
            colorize_mask_rgb(mask),
            caption="Predicted mask",
            width="stretch",
            clamp=True,
        )
        col_overlay.image(overlay, caption="Overlay", width="stretch", clamp=True)

    model_details = st.columns(4)
    model_details[0].metric("Architecture", loaded_model.spec.architecture)
    model_details[1].metric("Contrastive weight", loaded_model.spec.contrastive_weight)
    model_details[2].metric("Device", str(device))
    model_details[3].metric("Trainable params", f"{loaded_model.trainable_parameters:,}")

    result_details = st.columns(3)
    result_details[0].metric("Foreground pixels", f"{foreground_pixels:,}")
    result_details[1].metric("Mean Dice", "n/a" if mean_dice is None else f"{mean_dice:.4f}")
    result_details[2].metric("Mean IoU", "n/a" if mean_iou is None else f"{mean_iou:.4f}")

    with st.expander("Legend", expanded=True):
        for name, color in legend_rows():
            st.markdown(
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"background:{color};border:1px solid #888;margin-right:8px'></span>{name}",
                unsafe_allow_html=True,
            )

    with st.expander("Prediction details", expanded=False):
        st.dataframe(pred_rows, hide_index=True)
        if metric_rows:
            st.dataframe(metric_rows, hide_index=True)
        elif ground_truth_mask is not None:
            st.caption("No foreground class is present in ground truth or prediction.")

    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Download class-id mask",
        data=class_id_mask_to_png_bytes(mask),
        file_name="predicted_class_id_mask.png",
        mime="image/png",
    )
    download_cols[1].download_button(
        "Download colored mask",
        data=rgb_to_png_bytes(colorize_mask_rgb(mask)),
        file_name="predicted_colored_mask.png",
        mime="image/png",
    )
    download_cols[2].download_button(
        "Download overlay",
        data=image_to_png_bytes(overlay),
        file_name="prediction_overlay.png",
        mime="image/png",
    )

    st.caption(f"Checkpoint: {loaded_model.checkpoint_path} ({loaded_model.source})")


if __name__ == "__main__":
    main()

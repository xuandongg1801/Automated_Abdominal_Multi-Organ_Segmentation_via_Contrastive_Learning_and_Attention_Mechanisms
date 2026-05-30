from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
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
from app.model_registry import DEFAULT_MODEL_LABEL, MODEL_SPECS


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


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8), mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)).save(
        buffer,
        format="PNG",
    )
    return buffer.getvalue()


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

    with st.sidebar:
        model_labels = list(MODEL_SPECS)
        model_label = st.selectbox(
            "Model",
            model_labels,
            index=model_labels.index(DEFAULT_MODEL_LABEL),
        )
        device_name = st.radio("Device", ["cpu", "cuda"], horizontal=True)
        use_hu_window = st.toggle("HU window [-125, 275]", value=True)
        alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.5, 0.05)

        st.caption("Kaggle handle")
        st.code(MODEL_SPECS[model_label].handle, language=None)

    uploaded_file = st.file_uploader(
        "Upload CT slice",
        type=("png", "jpg", "jpeg", "tif", "tiff", "npy", "npz"),
    )

    if uploaded_file is None:
        st.info("Upload a CT slice or processed Synapse array to run segmentation.")
        return

    arrays = read_uploaded_arrays(uploaded_file)
    image_key = default_image_key(arrays)
    if len(arrays) > 1:
        image_key = st.selectbox("Array key", list(arrays), index=list(arrays).index(image_key))

    array = arrays[image_key]
    count = slice_count(array)
    slice_index = 0
    if count > 1:
        slice_index = st.slider("Slice", 0, count - 1, min(count // 2, count - 1))

    image = select_slice(array, slice_index)
    label_key = default_label_key(arrays, image_key)
    label_values: np.ndarray | None = None
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
        st.error(f"Inference failed: {type(exc).__name__}: {exc}")
        st.stop()

    overlay = make_overlay(display_image, mask, alpha=alpha)

    col_image, col_mask, col_overlay = st.columns(3)
    col_image.image(display_image, caption="Input", use_container_width=True, clamp=True)
    col_mask.image(colorize_mask_rgb(mask), caption="Predicted mask", use_container_width=True, clamp=True)
    col_overlay.image(overlay, caption="Overlay", use_container_width=True, clamp=True)

    details = st.columns(4)
    details[0].metric("Architecture", loaded_model.spec.architecture)
    details[1].metric("Contrastive weight", loaded_model.spec.contrastive_weight)
    details[2].metric("Device", str(device))
    details[3].metric("Trainable params", f"{loaded_model.trainable_parameters:,}")

    with st.expander("Legend", expanded=True):
        for name, color in legend_rows():
            st.markdown(
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"background:{color};border:1px solid #888;margin-right:8px'></span>{name}",
                unsafe_allow_html=True,
            )

    st.download_button(
        "Download mask",
        data=mask_to_png_bytes(mask),
        file_name="predicted_mask.png",
        mime="image/png",
    )
    st.download_button(
        "Download overlay",
        data=image_to_png_bytes(overlay),
        file_name="prediction_overlay.png",
        mime="image/png",
    )

    st.caption(f"Checkpoint: {loaded_model.checkpoint_path} ({loaded_model.source})")


if __name__ == "__main__":
    main()

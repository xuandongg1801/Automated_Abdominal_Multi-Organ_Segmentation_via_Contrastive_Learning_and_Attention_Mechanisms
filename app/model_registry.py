from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Deployment metadata for one Kaggle Model variation."""

    label: str
    architecture: str
    handle: str
    checkpoint_name: str
    contrastive_weight: float
    local_dir_env: str
    transformer_model_name: str = "vit_base_patch16_224.augreg_in21k"

    @property
    def local_dir(self) -> str:
        return os.getenv(self.local_dir_env, "").strip()


MODEL_SPECS: dict[str, ModelSpec] = {
    "ResNet-UNet baseline (cw0)": ModelSpec(
        label="ResNet-UNet baseline (cw0)",
        architecture="ResNetUNet",
        handle="tensura3607/abdominal-multi-organ-segmentation/pyTorch/resnet-unet",
        checkpoint_name="best_resnet_unet.pt",
        contrastive_weight=0.0,
        local_dir_env="RESNET_UNET_MODEL_DIR",
    ),
    "ResNet-UNet contrastive 0.01": ModelSpec(
        label="ResNet-UNet contrastive 0.01",
        architecture="ResNetUNet",
        handle="tensura3607/abdominal-multi-organ-segmentation/pyTorch/resnet-unet-cw001",
        checkpoint_name="best_resnet_unet.pt",
        contrastive_weight=0.01,
        local_dir_env="RESNET_UNET_CW001_MODEL_DIR",
    ),
    "ResNet-UNet contrastive 0.03": ModelSpec(
        label="ResNet-UNet contrastive 0.03",
        architecture="ResNetUNet",
        handle="tensura3607/abdominal-multi-organ-segmentation/pyTorch/resnet-unet-cw003",
        checkpoint_name="best_resnet_unet.pt",
        contrastive_weight=0.03,
        local_dir_env="RESNET_UNET_CW003_MODEL_DIR",
    ),
    "ResNet-UNet contrastive 0.05": ModelSpec(
        label="ResNet-UNet contrastive 0.05",
        architecture="ResNetUNet",
        handle="tensura3607/abdominal-multi-organ-segmentation/pyTorch/resnet-unet-cw005",
        checkpoint_name="best_resnet_unet.pt",
        contrastive_weight=0.05,
        local_dir_env="RESNET_UNET_CW005_MODEL_DIR",
    ),
    "TransUNet baseline (cw0)": ModelSpec(
        label="TransUNet baseline (cw0)",
        architecture="TransUNet",
        handle="tensura3607/abdominal-multi-organ-segmentation/pyTorch/transunet",
        checkpoint_name="best_transunet.pt",
        contrastive_weight=0.0,
        local_dir_env="TRANSUNET_MODEL_DIR",
    ),
    "TransUNet contrastive 0.01": ModelSpec(
        label="TransUNet contrastive 0.01",
        architecture="TransUNet",
        handle="tensura3607/abdominal-multi-organ-segmentation/pyTorch/transunet-cw001",
        checkpoint_name="best_transunet.pt",
        contrastive_weight=0.01,
        local_dir_env="TRANSUNET_CW001_MODEL_DIR",
    ),
    "TransUNet contrastive 0.03": ModelSpec(
        label="TransUNet contrastive 0.03",
        architecture="TransUNet",
        handle="tensura3607/abdominal-multi-organ-segmentation/pyTorch/transunet-cw003",
        checkpoint_name="best_transunet.pt",
        contrastive_weight=0.03,
        local_dir_env="TRANSUNET_CW003_MODEL_DIR",
    ),
    "TransUNet contrastive 0.05": ModelSpec(
        label="TransUNet contrastive 0.05",
        architecture="TransUNet",
        handle="tensura3607/abdominal-multi-organ-segmentation/pyTorch/transunet-cw005",
        checkpoint_name="best_transunet.pt",
        contrastive_weight=0.05,
        local_dir_env="TRANSUNET_CW005_MODEL_DIR",
    ),
}


DEFAULT_MODEL_LABEL = "TransUNet contrastive 0.01"


CLASS_NAMES = {
    0: "Background",
    1: "Spleen",
    2: "Right kidney",
    3: "Left kidney",
    4: "Gallbladder",
    5: "Liver",
    6: "Stomach",
    7: "Aorta",
    8: "Pancreas",
}

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from torchvision import datasets, models, transforms

MODEL_NAMES = (
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
    "swin_t",
    "swin_v2_t",
    "maxvit_t",
    "convnext_t",
    "convnext_s",
    "swin_v2_s",
    "late_fusion_t",
    "late_fusion_weighted_t",
    "late_fusion_attention_t",
)


RESNET_BUILDERS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights),
    "resnet34": (models.resnet34, models.ResNet34_Weights),
    "resnet50": (models.resnet50, models.ResNet50_Weights),
    "resnet101": (models.resnet101, models.ResNet101_Weights),
    "resnet152": (models.resnet152, models.ResNet152_Weights),
}

DENSENET_BUILDERS = {
    "densenet121": (models.densenet121, models.DenseNet121_Weights),
    "densenet161": (models.densenet161, models.DenseNet161_Weights),
    "densenet169": (models.densenet169, models.DenseNet169_Weights),
    "densenet201": (models.densenet201, models.DenseNet201_Weights),
}

EFFICIENTNET_BUILDERS = {
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights),
    "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights),
    "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights),
    "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights),
    "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights),
    "efficientnet_b5": (models.efficientnet_b5, models.EfficientNet_B5_Weights),
    "efficientnet_b6": (models.efficientnet_b6, models.EfficientNet_B6_Weights),
    "efficientnet_b7": (models.efficientnet_b7, models.EfficientNet_B7_Weights),
}

EFFICIENTNET_V2_BUILDERS = {
    "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights),
    "efficientnet_v2_m": (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights),
    "efficientnet_v2_l": (models.efficientnet_v2_l, models.EfficientNet_V2_L_Weights),
}

CONVNEXT_BUILDERS = {
    "convnext_t": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights),
    "convnext_s": (models.convnext_small, models.ConvNeXt_Small_Weights),
}

SWINV2_BUILDERS = {
    "swin_v2_t": (models.swin_v2_t, models.Swin_V2_T_Weights),
    "swin_v2_s": (models.swin_v2_s, models.Swin_V2_S_Weights),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare TB classification models on train/ and val/."
    )
    parser.add_argument("--train-dir", type=str, default="train")
    parser.add_argument("--val-dir", type=str, default="val")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_NAMES),
        choices=list(MODEL_NAMES),
        help="Models to train and compare.",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use torchvision pretrained weights when available.",
        default=True
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra training diagnostics.",
        default=True
    )
    parser.add_argument(
        "--use-dicom",
        action="store_true",
        help="Load DICOM files directly instead of JPG/PNG images.",
        default=False
    )
    parser.add_argument(
        "--use-preprocessing",
        action="store_true",
        help="Apply grayscale, CLAHE, RGB conversion, and per-image standardization.",
        default=False  # Changed default to False for transformer stability
    )
    parser.add_argument(
        "--use-clahe-only",
        action="store_true",
        help="Apply CLAHE preprocessing but keep ImageNet normalization (recommended for transformers).",
        default=False  # Set to True as default for better transformer performance
    )
    parser.add_argument(
        "--fusion-convnext-model",
        type=str,
        default="convnext_t",
        choices=["convnext_t", "convnext_s"],
        help="ConvNeXt backbone used by the late-fusion models.",
    )
    parser.add_argument(
        "--fusion-swin-v2-model",
        type=str,
        default="swin_v2_t",
        choices=["swin_v2_t", "swin_v2_s"],
        help="Swin V2 backbone used by the late-fusion models.",
    )
    parser.add_argument(
        "--fusion-freeze-epochs",
        type=int,
        default=3,
        help="Freeze the late-fusion backbones for this many initial epochs.",
    )
    parser.add_argument(
        "--fusion-unfreeze-lr-factor",
        type=float,
        default=0.5,
        help="Multiply the original learning rate by this factor after late-fusion backbones unfreeze.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=4,
        help="Stop training if validation F1 does not improve for this many epochs.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation F1 improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        default=False,
        help="Use a deterministic 25% subset of train and val for quick hyperparameter checks.",
    )
    parser.add_argument(
        "--train-test-split",
        type=float,
        default=0.2,
        help="Fraction of the training set to reserve as a held-out test set.",
    )
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        default=False,
        help="Use nn.DataParallel across all visible CUDA devices.",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Maximum gradient norm for gradient clipping.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Number of warmup epochs for learning rate scheduler.",
    )
    parser.add_argument(
        "--transformer-lr",
        type=float,
        default=5e-5,
        help="Learning rate specifically for transformer models (overrides --lr for transformers).",
    )
    parser.add_argument(
        "--use-laf",
        action="store_true",
        default=False,
        help=(
            "Insert the LAFAN-Net Large Adaptive Filter (LAF) block on the final "
            "spatial feature map, before global pooling. Applies to convnext_t/s, "
            "swin_t, swin_v2_t/s, and all late_fusion_* models."
        ),
    )
    parser.add_argument(
        "--laf-kernel-size1",
        type=int,
        default=7,
        help="Kernel size of the first depthwise conv in the LAF block.",
    )
    parser.add_argument(
        "--laf-kernel-size2",
        type=int,
        default=11,
        help="Kernel size of the second depthwise conv in the LAF block.",
    )
    parser.add_argument(
        "--use-alignnorm",
        action="store_true",
        default=False,
        help=(
            "Replace the final normalization before the classifier head with the "
            "LAFAN-Net AlignNorm layer, which fights representation oversmoothing. "
            "Applies to convnext_t/s, swin_t, swin_v2_t/s, and all late_fusion_* models."
        ),
    )
    parser.add_argument(
        "--alignnorm-temperature",
        type=float,
        default=0.1,
        help="Softmax temperature for AlignNorm's batch similarity matrix.",
    )
    parser.add_argument(
        "--alignnorm-scale",
        type=float,
        default=0.5,
        help="Scale applied to AlignNorm's similarity-weighted negative term.",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CLAHETransform:
    """Apply CLAHE and Gaussian blur, keeping the image in RGB format."""
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image: Image.Image) -> Image.Image:
        import cv2
        import numpy as np

        if image.mode != "L":
            image = image.convert("L")

        arr = np.array(image)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_grid_size, self.tile_grid_size),
        )
        arr = clahe.apply(arr)
        return Image.fromarray(arr, mode="L").convert("RGB")


class PerImageStandardize:
    """Standardize each image to zero mean and unit variance."""
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-8)


class LargeAdaptiveFilter(nn.Module):
    """
    Large Adaptive Filter (LAF) block, following LAFAN-Net (Lu, Zhu, Zhang,
    Yao, 2025 -- "Tuberculosis and pneumonia diagnosis in chest X-rays by
    large adaptive filter and aligning normalized network with
    report-guided multi-level alignment", Eng. Appl. Artif. Intell. 158).

    Two large-kernel depthwise convolutions are applied sequentially so the
    block sees two different effective receptive fields from the same
    input (the paper motivates this as: smaller/first-stage kernels pick up
    fine detail such as nodules, larger/second-stage kernels pick up
    broader patterns such as consolidation/infiltrates). The two resulting
    feature maps are concatenated, channel-pooled (max + average) into a
    2-channel spatial descriptor, and passed through a small conv +
    sigmoid to produce two per-pixel attention maps (one per scale). Those
    attention maps adaptively re-weight and fuse the two scales, and a
    final 1x1 conv turns the fused signal into a per-channel gate that is
    applied multiplicatively to the original input (Eqs. 12-16 in the
    paper).

    This block only assumes a (B, C, H, W) spatial feature map, so it is
    architecture-agnostic: it is used below on ConvNeXt's stage-5 feature
    map and on Swin/Swin-V2's stage-4 feature map (after permuting from the
    NHWC layout torchvision uses internally to NCHW).
    """

    def __init__(self, channels: int, kernel_size1: int = 7, kernel_size2: int = 11):
        super().__init__()
        self.dwc1 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size1,
            padding=kernel_size1 // 2, groups=channels, bias=False,
        )
        self.dwc2 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size2,
            padding=kernel_size2 // 2, groups=channels, bias=False,
        )
        self.spatial_attn = nn.Conv2d(2, 2, kernel_size=7, padding=3, bias=True)
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x1 = self.dwc1(x)
        x2 = self.dwc2(x1)
        cat = torch.cat([x1, x2], dim=1)             # (B, 2C, H, W), Eq. 12
        s_max = torch.amax(cat, dim=1, keepdim=True)  # channel-wise max pool
        s_avg = torch.mean(cat, dim=1, keepdim=True)  # channel-wise avg pool
        s = torch.cat([s_max, s_avg], dim=1)          # (B, 2, H, W), Eq. 13
        attn = torch.sigmoid(self.spatial_attn(s))    # (B, 2, H, W), Eq. 14
        a1, a2 = attn[:, 0:1], attn[:, 1:2]
        weighted = a1 * x1 + a2 * x2
        gate = self.fuse(weighted)                    # Eq. 15
        return x * gate                                # Eq. 16


class AlignNorm(nn.Module):
    """
    AlignNorm from LAFAN-Net: a normalization layer designed to counteract
    representation oversmoothing/dimensional collapse (Eqs. 17-20 in the
    paper). For a batch of feature vectors X in R^(B x D):
      1. Intra-sample L2 normalization.
      2. A batch similarity matrix (softmax of scaled cosine similarities).
      3. A similarity-weighted "negative" mixture of the *original*
         (un-normalized) batch is subtracted from each sample, scaled by a
         fixed factor, which discourages samples from being pulled toward
         the batch mean and so preserves feature separation.
      4. A closing LayerNorm.

    Note: the similarity term is only meaningful for batch size >= 2 (there
    is nothing to contrast a single sample against), so this module falls
    back to a plain LayerNorm when it sees B < 2 -- this happens on the
    last, possibly-incomplete batch of an eval/test DataLoader (the train
    loader here uses drop_last=True precisely to avoid this case).
    """

    def __init__(self, dim: int, temperature: float = 0.1, scale: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.scale = scale
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        if x.size(0) < 2:
            return self.layer_norm(x)
        x_n = nn.functional.normalize(x, dim=1)                        # Eq. 17
        sim = torch.softmax((x_n @ x_n.t()) / self.temperature, dim=1)  # Eq. 18
        x_neg = sim @ x
        x_p = x - self.scale * x_neg                                    # Eq. 19
        return self.layer_norm(x_p)                                     # Eq. 20


def _get_swin_classifier_in_features(model: nn.Module) -> int:
    if hasattr(model, "head"):
        return model.head.in_features
    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        return model.heads.head.in_features
    raise AttributeError("Unsupported Swin classifier layout in this torchvision version")


def _extract_swin_features(model: nn.Module, x: torch.Tensor, laf: "LargeAdaptiveFilter | None" = None) -> torch.Tensor:
    x = model.features(x)
    if hasattr(model, "norm"):
        x = model.norm(x)
    if x.ndim == 4:
        x = x.permute(0, 3, 1, 2).contiguous()
    if laf is not None:
        x = laf(x)
    if hasattr(model, "avgpool"):
        x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x


class LateFusionConvNeXtSwinV2Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        fusion_mode: str = "concat",
        convnext_model_name: str = "convnext_t",
        swin_v2_model_name: str = "swin_v2_t",
        use_laf: bool = False,
        use_alignnorm: bool = False,
        laf_kernel_size1: int = 7,
        laf_kernel_size2: int = 11,
        alignnorm_temperature: float = 0.1,
        alignnorm_scale: float = 0.5,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.freeze_backbones = False
        self.convnext_model_name = convnext_model_name
        self.swin_v2_model_name = swin_v2_model_name
        self.use_laf = use_laf

        if convnext_model_name not in CONVNEXT_BUILDERS:
            raise ValueError(f"Unknown ConvNeXt backbone: {convnext_model_name}")
        if swin_v2_model_name not in SWINV2_BUILDERS:
            raise ValueError(f"Unknown Swin V2 backbone: {swin_v2_model_name}")

        conv_builder, conv_weight_enum = CONVNEXT_BUILDERS[convnext_model_name]
        swin_builder, swin_weight_enum = SWINV2_BUILDERS[swin_v2_model_name]
        conv_weights = conv_weight_enum.DEFAULT if pretrained else None
        swin_weights = swin_weight_enum.DEFAULT if pretrained else None

        self.convnext = conv_builder(weights=conv_weights)
        self.swin_v2 = swin_builder(weights=swin_weights)

        conv_features = self.convnext.classifier[-1].in_features
        swin_features = _get_swin_classifier_in_features(self.swin_v2)

        # Large Adaptive Filter blocks operate on each backbone's own spatial
        # feature map (before global pooling), so one is needed per backbone
        # since ConvNeXt and Swin-V2 generally have different channel counts.
        self.conv_laf = (
            LargeAdaptiveFilter(conv_features, laf_kernel_size1, laf_kernel_size2)
            if use_laf else None
        )
        self.swin_laf = (
            LargeAdaptiveFilter(swin_features, laf_kernel_size1, laf_kernel_size2)
            if use_laf else None
        )

        def _make_fusion_head(fusion_features: int) -> nn.Sequential:
            norm_layer = (
                AlignNorm(fusion_features, alignnorm_temperature, alignnorm_scale)
                if use_alignnorm else nn.LayerNorm(fusion_features)
            )
            return nn.Sequential(
                norm_layer,
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(fusion_features, num_classes),
            )

        if fusion_mode == "concat":
            fusion_features = conv_features + swin_features
            self.alpha = None
            self.beta = None
            self.conv_proj = None
            self.swin_proj = None
            self.fusion_dropout = nn.Dropout(0.35)
            self.fusion_head = _make_fusion_head(fusion_features)
        elif fusion_mode == "weighted":
            fusion_features = min(conv_features, swin_features)
            self.raw_alpha = nn.Parameter(torch.tensor(0.5))
            self.conv_proj = nn.Linear(conv_features, fusion_features) if conv_features != fusion_features else nn.Identity()
            self.swin_proj = nn.Linear(swin_features, fusion_features) if swin_features != fusion_features else nn.Identity()
            self.fusion_dropout = nn.Dropout(0.35)
            self.fusion_head = _make_fusion_head(fusion_features)
        elif fusion_mode == "attention":
            fusion_features = min(conv_features, swin_features)
            self.raw_alpha = nn.Parameter(torch.tensor(0.5))
            self.raw_beta = nn.Parameter(torch.tensor(0.5))
            self.conv_proj = nn.Linear(conv_features, fusion_features) if conv_features != fusion_features else nn.Identity()
            self.swin_proj = nn.Linear(swin_features, fusion_features) if swin_features != fusion_features else nn.Identity()
            self.fusion_dropout = nn.Dropout(0.35)
            self.fusion_head = _make_fusion_head(fusion_features)
        else:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")

    def set_backbones_trainable(self, trainable: bool) -> None:
        self.freeze_backbones = not trainable
        self.convnext.requires_grad_(trainable)
        self.swin_v2.requires_grad_(trainable)
        if not trainable:
            self.convnext.eval()
            self.swin_v2.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Only the backbone forward passes are wrapped in no_grad when
        # frozen (their params already have requires_grad=False, courtesy
        # of set_backbones_trainable). The LAF blocks are kept outside that
        # scope on purpose: they are lightweight adapters with their own
        # learnable parameters, and gradients still flow into those
        # parameters even though the backbone activations feeding them
        # carry requires_grad=False -- autograd only needs the LAF's own
        # weights to require grad to update them. Nesting the LAF call
        # inside no_grad, as a naive edit might do, would silently prevent
        # it from ever training during the frozen-backbone warmup epochs.
        if self.freeze_backbones:
            with torch.no_grad():
                conv_map = self.convnext.features(images)
                swin_map = self.swin_v2.features(images)
                if hasattr(self.swin_v2, "norm"):
                    swin_map = self.swin_v2.norm(swin_map)
        else:
            conv_map = self.convnext.features(images)
            swin_map = self.swin_v2.features(images)
            if hasattr(self.swin_v2, "norm"):
                swin_map = self.swin_v2.norm(swin_map)

        if swin_map.ndim == 4:
            swin_map = swin_map.permute(0, 3, 1, 2).contiguous()

        if self.conv_laf is not None:
            conv_map = self.conv_laf(conv_map)
        if self.swin_laf is not None:
            swin_map = self.swin_laf(swin_map)

        conv_feat = self.convnext.avgpool(conv_map)
        conv_feat = torch.flatten(conv_feat, 1)

        if hasattr(self.swin_v2, "avgpool"):
            swin_feat = self.swin_v2.avgpool(swin_map)
        else:
            swin_feat = swin_map
        swin_feat = torch.flatten(swin_feat, 1)

        if self.fusion_mode == "concat":
            fused = torch.cat([conv_feat, swin_feat], dim=1)
        elif self.fusion_mode == "weighted":
            conv_feat = self.conv_proj(conv_feat)
            swin_feat = self.swin_proj(swin_feat)
            conv_feat = nn.functional.normalize(conv_feat, dim=1)
            swin_feat = nn.functional.normalize(swin_feat, dim=1)
            alpha = torch.sigmoid(self.raw_alpha)
            beta = 1.0 - alpha
            fused = alpha * conv_feat + beta * swin_feat
        else:
            conv_feat = self.conv_proj(conv_feat)
            swin_feat = self.swin_proj(swin_feat)
            conv_feat = nn.functional.normalize(conv_feat, dim=1)
            swin_feat = nn.functional.normalize(swin_feat, dim=1)
            alpha = torch.sigmoid(self.raw_alpha)
            beta = torch.sigmoid(self.raw_beta)
            fused = alpha * conv_feat + beta * swin_feat
        fused = self.fusion_dropout(fused)
        return self.fusion_head(fused)


class ConvNeXtWithLAFAN(nn.Module):
    """
    Wraps a torchvision ConvNeXt so that, optionally:
      - a LargeAdaptiveFilter runs on the final (B, C, H, W) feature map
        from model.features, before global average pooling; and/or
      - an AlignNorm replaces the LayerNorm2d that torchvision's ConvNeXt
        classifier normally applies to the pooled (B, C, 1, 1) features.
    Reimplementing forward() this way (rather than editing model.classifier
    in place) keeps the original pretrained ConvNeXt submodules untouched,
    which matters if you ever want to load a checkpoint back into a plain
    torchvision ConvNeXt for comparison.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        use_laf: bool = False,
        use_alignnorm: bool = False,
        laf_kernel_size1: int = 7,
        laf_kernel_size2: int = 11,
        alignnorm_temperature: float = 0.1,
        alignnorm_scale: float = 0.5,
    ):
        super().__init__()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        in_features = base_model.classifier[-1].in_features

        self.laf = (
            LargeAdaptiveFilter(in_features, laf_kernel_size1, laf_kernel_size2)
            if use_laf else None
        )
        self.norm = (
            AlignNorm(in_features, alignnorm_temperature, alignnorm_scale)
            if use_alignnorm else base_model.classifier[0]  # original LayerNorm2d
        )
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.laf is not None:
            x = self.laf(x)
        x = self.avgpool(x)
        if isinstance(self.norm, AlignNorm):
            x = torch.flatten(x, 1)
            x = self.norm(x)
        else:
            x = self.norm(x)  # LayerNorm2d expects (B, C, 1, 1)
            x = torch.flatten(x, 1)
        return self.head(x)


class SwinWithLAFAN(nn.Module):
    """
    Wraps a torchvision Swin / Swin-V2 model so that, optionally:
      - a LargeAdaptiveFilter runs on the final spatial feature map (after
        permuting torchvision's internal NHWC layout to NCHW), before
        global average pooling; and/or
      - an AlignNorm is applied to the pooled feature vector before the
        classification head.
    Works for swin_t, swin_v2_t, and swin_v2_s since they share the same
    features / norm / avgpool / head structure in torchvision.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        use_laf: bool = False,
        use_alignnorm: bool = False,
        laf_kernel_size1: int = 7,
        laf_kernel_size2: int = 11,
        alignnorm_temperature: float = 0.1,
        alignnorm_scale: float = 0.5,
    ):
        super().__init__()
        self.base_features = base_model.features
        self.base_norm = getattr(base_model, "norm", None)
        self.avgpool = base_model.avgpool
        in_features = _get_swin_classifier_in_features(base_model)

        self.laf = (
            LargeAdaptiveFilter(in_features, laf_kernel_size1, laf_kernel_size2)
            if use_laf else None
        )
        self.align_norm = (
            AlignNorm(in_features, alignnorm_temperature, alignnorm_scale)
            if use_alignnorm else None
        )
        self.head = nn.Linear(in_features, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_features(x)
        if self.base_norm is not None:
            x = self.base_norm(x)
        if x.ndim == 4:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.laf is not None:
            x = self.laf(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.align_norm is not None:
            x = self.align_norm(x)
        return self.head(x)


def build_transforms(img_size: int, use_clahe: bool, use_clahe_only: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build train and validation transforms.
    
    Args:
        img_size: Target image size
        use_clahe: If True, apply CLAHE + per-image standardization (good for CNNs)
        use_clahe_only: If True, apply CLAHE but keep ImageNet normalization (good for transformers)
    """
    shared_prefix = []
    if use_clahe or use_clahe_only:
        shared_prefix.append(CLAHETransform())

    if use_clahe and not use_clahe_only:
        # Full preprocessing with per-image standardization
        train_tfms = transforms.Compose(
            shared_prefix
            + [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                PerImageStandardize(),
            ]
        )
        val_tfms = transforms.Compose(
            shared_prefix
            + [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                PerImageStandardize(),
            ]
        )
    else:
        # Standard augmentation with ImageNet normalization (compatible with transformers)
        train_tfms = transforms.Compose(
            shared_prefix
            + [
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        val_tfms = transforms.Compose(
            shared_prefix
            + [
                transforms.Resize(int(img_size * 1.14)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return train_tfms, val_tfms


class DicomFolderDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not self.classes:
            raise ValueError(f"No class folders found under {root}")
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = self._make_samples()
        self.targets = [target for _, target in self.samples]

    def _make_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in {".dcm", ".dicom"}:
                    samples.append((str(path), self.class_to_idx[class_name]))
        if not samples:
            raise ValueError(f"No DICOM files found under {self.root}")
        return samples

    def _load_dicom(self, path: str) -> Image.Image:
        import numpy as np
        import pydicom

        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
        if photometric == "MONOCHROME1":
            arr = arr.max() - arr

        lo, hi = np.percentile(arr, (1.0, 99.0))
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            arr = np.zeros_like(arr, dtype=np.uint8)
        else:
            arr = np.clip(arr, lo, hi)
            arr = ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)

        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        if arr.ndim == 3 and arr.shape[-1] in {3, 4}:
            return Image.fromarray(arr).convert("RGB")
        raise ValueError(f"Unsupported DICOM pixel shape {arr.shape} for {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self._load_dicom(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_datasets(
    train_dir: str, 
    val_dir: str, 
    img_size: int, 
    use_dicom: bool, 
    use_preprocessing: bool,
    use_clahe_only: bool = False
):
    train_tfms, val_tfms = build_transforms(img_size, use_preprocessing, use_clahe_only)
    if use_dicom:
        train_ds = DicomFolderDataset(train_dir, transform=train_tfms)
        val_ds = DicomFolderDataset(val_dir, transform=val_tfms)
    else:
        train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
        val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)
    if train_ds.classes != val_ds.classes:
        raise ValueError(
            f"Train and validation class folders differ: {train_ds.classes} vs {val_ds.classes}"
        )
    return train_ds, val_ds


def build_loaders(
    train_ds,
    val_ds,
    test_ds,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # Avoid batch size 1 for BatchNorm/LayerNorm stability
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def _dataset_targets(dataset) -> List[int]:
    if isinstance(dataset, Subset):
        parent_targets = _dataset_targets(dataset.dataset)
        return [parent_targets[i] for i in dataset.indices]
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    raise AttributeError("Dataset does not expose targets for class-weight computation")


def _dataset_classes(dataset) -> List[str]:
    if isinstance(dataset, Subset):
        return _dataset_classes(dataset.dataset)
    if hasattr(dataset, "classes"):
        return list(dataset.classes)
    raise AttributeError("Dataset does not expose classes")


def _dataset_samples(dataset) -> List[Tuple[str, int]]:
    if isinstance(dataset, Subset):
        parent_samples = _dataset_samples(dataset.dataset)
        return [parent_samples[i] for i in dataset.indices]
    if hasattr(dataset, "samples"):
        return list(dataset.samples)
    raise AttributeError("Dataset does not expose samples")


def _dataset_root_indices(dataset) -> List[int]:
    if isinstance(dataset, Subset):
        parent_indices = _dataset_root_indices(dataset.dataset)
        return [parent_indices[i] for i in dataset.indices]
    return list(range(len(dataset)))


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def compute_class_weights(dataset: datasets.ImageFolder, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(_dataset_targets(dataset)), minlength=num_classes).float()
    weights = counts.sum() / (counts * num_classes)
    return weights


def _stratified_sample_indices(dataset, fraction: float, seed: int) -> List[int]:
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    targets = _dataset_targets(dataset)
    by_class: Dict[int, List[int]] = {}
    for index, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(index)

    generator = torch.Generator().manual_seed(seed)
    chosen: List[int] = []
    for class_indices in by_class.values():
        perm = torch.randperm(len(class_indices), generator=generator).tolist()
        shuffled = [class_indices[i] for i in perm]
        take = max(1, int(round(len(shuffled) * fraction))) if len(shuffled) > 1 else len(shuffled)
        take = min(take, len(shuffled))
        chosen.extend(shuffled[:take])

    return sorted(chosen)


def make_stratified_subset(dataset, fraction: float = 0.25, seed: int = 42) -> Subset:
    if fraction == 1.0:
        return Subset(dataset, list(range(len(dataset))))
    return Subset(dataset, _stratified_sample_indices(dataset, fraction, seed))


def make_mini_subset(dataset, fraction: float = 0.25, seed: int = 42) -> Subset:
    return make_stratified_subset(dataset, fraction=fraction, seed=seed)


def split_dataset_stratified(dataset, test_fraction: float = 0.2, seed: int = 42) -> Tuple[Subset, Subset]:
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    targets = _dataset_targets(dataset)
    by_class: Dict[int, List[int]] = {}
    for index, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(index)

    generator = torch.Generator().manual_seed(seed)
    keep_indices: List[int] = []
    test_indices: List[int] = []

    for class_indices in by_class.values():
        perm = torch.randperm(len(class_indices), generator=generator).tolist()
        shuffled = [class_indices[i] for i in perm]
        test_count = int(round(len(shuffled) * test_fraction))
        if len(shuffled) > 1:
            test_count = max(1, min(test_count, len(shuffled) - 1))
        test_indices.extend(shuffled[:test_count])
        keep_indices.extend(shuffled[test_count:])

    return Subset(dataset, sorted(keep_indices)), Subset(dataset, sorted(test_indices))


def save_test_split(
    output_dir: Path,
    train_dataset,
    test_dataset,
    train_dir: str,
    train_test_split: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "test_split.json"
    samples = _dataset_samples(test_dataset)
    root_indices = _dataset_root_indices(test_dataset)
    payload = {
        "train_dir": train_dir,
        "train_test_split": train_test_split,
        "sample_count": len(samples),
        "classes": _dataset_classes(train_dataset),
        "root_indices": root_indices,
        "samples": [
            {"index": idx, "path": path, "target": target}
            for idx, (path, target) in zip(root_indices, samples)
        ],
    }
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return split_path


def is_transformer_model(model_name: str) -> bool:
    """Check if the model is a transformer-based architecture."""
    transformer_models = {"swin_t", "swin_v2_t", "swin_v2_s", "maxvit_t"}
    return model_name in transformer_models


def get_model(
    name: str,
    num_classes: int,
    pretrained: bool,
    img_size: int,
    fusion_convnext_model: str = "convnext_t",
    fusion_swin_v2_model: str = "swin_v2_t",
    use_laf: bool = False,
    use_alignnorm: bool = False,
    laf_kernel_size1: int = 7,
    laf_kernel_size2: int = 11,
    alignnorm_temperature: float = 0.1,
    alignnorm_scale: float = 0.5,
) -> nn.Module:
    if name in RESNET_BUILDERS:
        builder, weight_enum = RESNET_BUILDERS[name]
        weights = weight_enum.DEFAULT if pretrained else None
        model = builder(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name in DENSENET_BUILDERS:
        builder, weight_enum = DENSENET_BUILDERS[name]
        weights = weight_enum.DEFAULT if pretrained else None
        model = builder(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if name in EFFICIENTNET_BUILDERS:
        builder, weight_enum = EFFICIENTNET_BUILDERS[name]
        weights = weight_enum.DEFAULT if pretrained else None
        model = builder(weights=weights)
        classifier_in = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(classifier_in, num_classes)
        return model

    if name in EFFICIENTNET_V2_BUILDERS:
        builder, weight_enum = EFFICIENTNET_V2_BUILDERS[name]
        weights = weight_enum.DEFAULT if pretrained else None
        model = builder(weights=weights)
        classifier_in = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(classifier_in, num_classes)
        return model

    if name == "maxvit_t":
        if pretrained and img_size != 224:
            print(f"WARNING: maxvit_t pretrained weights expect 224x224 inputs. Setting img_size to 224.")
            img_size = 224
        if img_size % 224 != 0:
            raise ValueError(
                f"maxvit_t requires an input size that is a multiple of 224; got {img_size}. "
                "Try 224 or 448."
            )
        weights = models.MaxVit_T_Weights.DEFAULT if pretrained else None
        model = models.maxvit_t(weights=weights, input_size=(img_size, img_size))
        classifier_in = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(classifier_in, num_classes, bias=False)
        nn.init.trunc_normal_(model.classifier[-1].weight, std=0.02)
        return model

    if name == "swin_t":
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        if use_laf or use_alignnorm:
            return SwinWithLAFAN(
                model, num_classes,
                use_laf=use_laf, use_alignnorm=use_alignnorm,
                laf_kernel_size1=laf_kernel_size1, laf_kernel_size2=laf_kernel_size2,
                alignnorm_temperature=alignnorm_temperature, alignnorm_scale=alignnorm_scale,
            )
        if hasattr(model, "head"):
            classifier_in = model.head.in_features
            model.head = nn.Linear(classifier_in, num_classes)
            nn.init.trunc_normal_(model.head.weight, std=0.02)
            if model.head.bias is not None:
                nn.init.zeros_(model.head.bias)
        elif hasattr(model, "heads") and hasattr(model.heads, "head"):
            classifier_in = model.heads.head.in_features
            model.heads.head = nn.Linear(classifier_in, num_classes)
            nn.init.trunc_normal_(model.heads.head.weight, std=0.02)
            if model.heads.head.bias is not None:
                nn.init.zeros_(model.heads.head.bias)
        else:
            raise AttributeError("Unsupported Swin-T classifier layout in this torchvision version")
        return model
    
    if name in SWINV2_BUILDERS:
        builder, weight_enum = SWINV2_BUILDERS[name]
        weights = weight_enum.DEFAULT if pretrained else None
        model = builder(weights=weights)
        if use_laf or use_alignnorm:
            return SwinWithLAFAN(
                model, num_classes,
                use_laf=use_laf, use_alignnorm=use_alignnorm,
                laf_kernel_size1=laf_kernel_size1, laf_kernel_size2=laf_kernel_size2,
                alignnorm_temperature=alignnorm_temperature, alignnorm_scale=alignnorm_scale,
            )
        if hasattr(model, "head"):
            classifier_in = model.head.in_features
            model.head = nn.Linear(classifier_in, num_classes)
            nn.init.trunc_normal_(model.head.weight, std=0.02)
            if model.head.bias is not None:
                nn.init.zeros_(model.head.bias)
        elif hasattr(model, "heads") and hasattr(model.heads, "head"):
            classifier_in = model.heads.head.in_features
            model.heads.head = nn.Linear(classifier_in, num_classes)
            nn.init.trunc_normal_(model.heads.head.weight, std=0.02)
            if model.heads.head.bias is not None:
                nn.init.zeros_(model.heads.head.bias)
        else:
            raise AttributeError("Unsupported Swin-V2 classifier layout in this torchvision version")
        return model
    
    if name in CONVNEXT_BUILDERS:
        builder, weight_enum = CONVNEXT_BUILDERS[name]
        weights = weight_enum.DEFAULT if pretrained else None
        model = builder(weights=weights)
        if use_laf or use_alignnorm:
            return ConvNeXtWithLAFAN(
                model, num_classes,
                use_laf=use_laf, use_alignnorm=use_alignnorm,
                laf_kernel_size1=laf_kernel_size1, laf_kernel_size2=laf_kernel_size2,
                alignnorm_temperature=alignnorm_temperature, alignnorm_scale=alignnorm_scale,
            )
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if name == "late_fusion_t":
        return LateFusionConvNeXtSwinV2Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            fusion_mode="concat",
            convnext_model_name=fusion_convnext_model,
            swin_v2_model_name=fusion_swin_v2_model,
            use_laf=use_laf,
            use_alignnorm=use_alignnorm,
            laf_kernel_size1=laf_kernel_size1,
            laf_kernel_size2=laf_kernel_size2,
            alignnorm_temperature=alignnorm_temperature,
            alignnorm_scale=alignnorm_scale,
        )

    if name == "late_fusion_weighted_t":
        return LateFusionConvNeXtSwinV2Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            fusion_mode="weighted",
            convnext_model_name=fusion_convnext_model,
            swin_v2_model_name=fusion_swin_v2_model,
            use_laf=use_laf,
            use_alignnorm=use_alignnorm,
            laf_kernel_size1=laf_kernel_size1,
            laf_kernel_size2=laf_kernel_size2,
            alignnorm_temperature=alignnorm_temperature,
            alignnorm_scale=alignnorm_scale,
        )

    if name == "late_fusion_attention_t":
        return LateFusionConvNeXtSwinV2Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            fusion_mode="attention",
            convnext_model_name=fusion_convnext_model,
            swin_v2_model_name=fusion_swin_v2_model,
            use_laf=use_laf,
            use_alignnorm=use_alignnorm,
            laf_kernel_size1=laf_kernel_size1,
            laf_kernel_size2=laf_kernel_size2,
            alignnorm_temperature=alignnorm_temperature,
            alignnorm_scale=alignnorm_scale,
        )
    
    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    tp = tn = fp = fn = 0
    all_probs = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        probs = torch.softmax(outputs, dim=1)[:, 1]   # probability of TB class
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        total += targets.size(0)
        correct += (preds == targets).sum().item()

        tp += ((preds == 1) & (targets == 1)).sum().item()
        tn += ((preds == 0) & (targets == 0)).sum().item()
        fp += ((preds == 1) & (targets == 0)).sum().item()
        fn += ((preds == 0) & (targets == 1)).sum().item()
    try:
        auroc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auroc = 0.0
    accuracy = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    avg_loss = total_loss / max(total, 1)
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "confusion": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step_scheduler_per_batch: bool,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    gradient_clip: float,
    freeze_backbones: bool = False,
):
    model.train()
    if freeze_backbones and hasattr(model, "convnext") and hasattr(model, "swin_v2"):
        model.convnext.eval()
        model.swin_v2.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type = "cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            
        scaler.step(optimizer)
        scaler.update()
        if step_scheduler_per_batch:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        total += targets.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()

    return {
        "loss": running_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    output_dir: Path,
    pretrained: bool,
    debug: bool,
    img_size: int,
    use_preprocessing: bool,
    use_clahe_only: bool,
    batch_size: int,
    mini: bool,
    train_test_split: float,
    test_split_path: str,
    fusion_convnext_model: str,
    fusion_swin_v2_model: str,
    fusion_freeze_epochs: int,
    fusion_unfreeze_lr_factor: float,
    data_parallel: bool,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    gradient_clip: float,
    warmup_epochs: int,
    transformer_lr: float,
    use_laf: bool = False,
    use_alignnorm: bool = False,
    laf_kernel_size1: int = 7,
    laf_kernel_size2: int = 11,
    alignnorm_temperature: float = 0.1,
    alignnorm_scale: float = 0.5,
):
    # Adjust learning rate for transformer models
    if is_transformer_model(model_name):
        effective_lr = transformer_lr
        if debug:
            print(f"[{model_name}] Using transformer-specific learning rate: {effective_lr}")
    else:
        effective_lr = lr
    
    model = get_model(
        model_name,
        num_classes=2,
        pretrained=pretrained,
        img_size=img_size,
        fusion_convnext_model=fusion_convnext_model,
        fusion_swin_v2_model=fusion_swin_v2_model,
        use_laf=use_laf,
        use_alignnorm=use_alignnorm,
        laf_kernel_size1=laf_kernel_size1,
        laf_kernel_size2=laf_kernel_size2,
        alignnorm_temperature=alignnorm_temperature,
        alignnorm_scale=alignnorm_scale,
    ).to(device)
    
    if data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    base_model = _unwrap_model(model)
    
    if model_name == "late_fusion_t":
        fusion_mode = "concat"
    elif model_name == "late_fusion_weighted_t":
        fusion_mode = "weighted"
    elif model_name == "late_fusion_attention_t":
        fusion_mode = "attention"
    else:
        fusion_mode = ""
    
    label_smoothing = 0.05 if model_name.startswith("late_fusion") else 0.0
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=label_smoothing,
    )
    
    effective_weight_decay = weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=effective_weight_decay)
    
    # Use warmup + cosine annealing scheduler
    if warmup_epochs > 0:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            total_iters=len(train_loader) * warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=len(train_loader) * (epochs - warmup_epochs)
        )
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[len(train_loader) * warmup_epochs]
        )
        step_scheduler_per_batch = True
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        step_scheduler_per_batch = False
    
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    early_stop_counter = 0
    stopped_epoch = epochs
    history: List[Dict[str, float]] = []

    if debug:
        print(f"\n[{model_name}] debug start")
        print(f"[{model_name}] device={device} pretrained={pretrained}")
        print(f"[{model_name}] img_size={img_size}")        
        print(f"[{model_name}] batch_size={batch_size}")
        print(f"[{model_name}] use_preprocessing={use_preprocessing}")
        print(f"[{model_name}] use_clahe_only={use_clahe_only}")
        print(f"[{model_name}] transformer_lr={transformer_lr}")
        print(f"[{model_name}] effective_lr={effective_lr}")
        print(f"[{model_name}] gradient_clip={gradient_clip}")
        print(f"[{model_name}] warmup_epochs={warmup_epochs}")
        print(f"[{model_name}] use_laf={use_laf} laf_kernel_size1={laf_kernel_size1} laf_kernel_size2={laf_kernel_size2}")
        print(f"[{model_name}] use_alignnorm={use_alignnorm} alignnorm_temperature={alignnorm_temperature} alignnorm_scale={alignnorm_scale}")
        print(f"[{model_name}] fusion_convnext_model={fusion_convnext_model}")
        print(f"[{model_name}] fusion_swin_v2_model={fusion_swin_v2_model}")
        print(f"[{model_name}] fusion_mode={fusion_mode}")
        print(f"[{model_name}] fusion_freeze_epochs={fusion_freeze_epochs}")
        print(f"[{model_name}] fusion_unfreeze_lr_factor={fusion_unfreeze_lr_factor}")
        print(f"[{model_name}] early_stopping_patience={early_stopping_patience}")
        print(f"[{model_name}] early_stopping_min_delta={early_stopping_min_delta}")
        print(f"[{model_name}] data_parallel={data_parallel}")
        print(f"[{model_name}] cuda_device_count={torch.cuda.device_count()}")
        print(f"[{model_name}] label_smoothing={label_smoothing}")
        print(f"[{model_name}] use_amp={use_amp}")
        print(f"[{model_name}] epochs={epochs}")
        print(f"[{model_name}] train_batches={len(train_loader)} val_batches={len(val_loader)} test_batches={len(test_loader)}")
        print(f"[{model_name}] train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)} test_samples={len(test_loader.dataset)}")
        print(f"[{model_name}] train_test_split={train_test_split}")
        print(f"[{model_name}] test_split_path={test_split_path}")
        print(f"[{model_name}] class_weights={class_weights.tolist()}")
        print(f"[{model_name}] model={model.__class__.__name__}")

        first_images, first_targets = next(iter(train_loader))
        print(f"[{model_name}] first_batch_images_shape={tuple(first_images.shape)}")
        print(f"[{model_name}] first_batch_targets_shape={tuple(first_targets.shape)}")
        print(f"[{model_name}] first_batch_targets={first_targets[:16].tolist()}")

    for epoch in range(1, epochs + 1):
        freeze_backbones = (
            model_name.startswith("late_fusion")
            and epoch <= fusion_freeze_epochs
            and hasattr(base_model, "set_backbones_trainable")
        )
        if model_name.startswith("late_fusion") and epoch == fusion_freeze_epochs + 1:
            old_lr = optimizer.param_groups[0]["lr"]
            switched_lr = effective_lr * fusion_unfreeze_lr_factor
            _set_optimizer_lr(optimizer, switched_lr)
            if debug:
                print(
                    f"[{model_name}] unfreezing backbones at epoch {epoch}; "
                    f"lr {old_lr:.6g} -> {switched_lr:.6g}"
                )
        if hasattr(base_model, "set_backbones_trainable"):
            base_model.set_backbones_trainable(not freeze_backbones)
        
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            step_scheduler_per_batch,
            scaler,
            device,
            use_amp,
            gradient_clip,
            freeze_backbones=freeze_backbones,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        if not step_scheduler_per_batch:
            scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": float(epoch),
            "lr": float(current_lr),
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["accuracy"]),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["accuracy"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_auroc": float(val_metrics["auroc"]),
        }
        history.append(row)

        print(
            f"[{model_name}] Epoch {epoch:02d}/{epochs} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_acc']:.4f} "
            f"val_f1={row['val_f1']:.4f} "
            f"val_auroc={row['val_auroc']:.4f}"
        )
        if debug:
            confusion = val_metrics["confusion"]
            print(
                f"[{model_name}] lr={current_lr:.6g} "
                f"val_precision={row['val_precision']:.4f} val_recall={row['val_recall']:.4f} "
                f"confusion={confusion} best_f1={best_f1:.4f}"
            )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            early_stop_counter = 0
            best_state = {
                "model_name": model_name,
                "epoch": epoch,
                "model_state": _unwrap_model(model).state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_metrics": val_metrics,
                "img_size": img_size,
                "use_preprocessing": use_preprocessing,
                "use_clahe_only": use_clahe_only,
                "batch_size": batch_size,
                "mini": mini,
                "data_parallel": data_parallel,
                "fusion_freeze_epochs": fusion_freeze_epochs,
                "fusion_unfreeze_lr_factor": fusion_unfreeze_lr_factor,
                "fusion_convnext_model": fusion_convnext_model,
                "fusion_swin_v2_model": fusion_swin_v2_model,
                "fusion_mode": fusion_mode,
                "test_split_path": test_split_path,
                "best_val_epoch": epoch,
                "best_val_f1": val_metrics["f1"],
            }
        else:
            if early_stopping_patience > 0 and val_metrics["f1"] < (best_f1 + early_stopping_min_delta):
                early_stop_counter += 1
                if early_stop_counter >= early_stopping_patience:
                    stopped_epoch = epoch
                    if debug:
                        print(
                            f"[{model_name}] early stopping triggered at epoch {epoch} "
                            f"(best_val_f1={best_f1:.4f}, patience={early_stopping_patience})"
                        )
                    break
            else:
                early_stop_counter = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{model_name}_best.pt"
    if best_state is not None:
        _unwrap_model(model).load_state_dict(best_state["model_state"])
        test_metrics = evaluate(model, test_loader, criterion, device)
        best_state["best_test_metrics"] = test_metrics
        best_state["train_test_split"] = train_test_split
        torch.save(best_state, checkpoint_path)
    else:
        test_metrics = {}

    history_path = output_dir / f"{model_name}_history.json"
    history_payload = {
        "model": model_name,
        "img_size": img_size,
        "use_preprocessing": use_preprocessing,
        "use_clahe_only": use_clahe_only,
        "batch_size": batch_size,
        "mini": mini,
        "data_parallel": data_parallel,
        "fusion_freeze_epochs": fusion_freeze_epochs,
        "fusion_unfreeze_lr_factor": fusion_unfreeze_lr_factor,
        "fusion_convnext_model": fusion_convnext_model,
        "fusion_swin_v2_model": fusion_swin_v2_model,
        "fusion_mode": fusion_mode,
        "train_test_split": train_test_split,
        "test_split_path": test_split_path,
        "total_training_samples": len(train_loader.dataset),
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "stopped_epoch": stopped_epoch,
        "best_val_epoch": best_epoch,
        "best_val_f1": best_f1,
        "best_test_metrics": test_metrics,
        "history": history,
    }
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    return {
        "model": model_name,
        "best_checkpoint": str(checkpoint_path),
        "best_val_f1": best_f1,
        "best_val_epoch": best_epoch,
        "img_size": img_size,
        "use_preprocessing": use_preprocessing,
        "use_clahe_only": use_clahe_only,
        "batch_size": batch_size,
        "mini": mini,
        "data_parallel": data_parallel,
        "fusion_freeze_epochs": fusion_freeze_epochs,
        "fusion_unfreeze_lr_factor": fusion_unfreeze_lr_factor,
        "fusion_convnext_model": fusion_convnext_model,
        "fusion_swin_v2_model": fusion_swin_v2_model,
        "fusion_mode": fusion_mode,
        "train_test_split": train_test_split,
        "test_split_path": test_split_path,
        "total_training_samples": len(train_loader.dataset),
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "stopped_epoch": stopped_epoch,
        "test_metrics": test_metrics,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "history": history,
    }


def main() -> None:
    args = parse_args()
    
    # Handle maxvit_t image size requirement
    if "maxvit_t" in args.models and args.img_size != 224:
        print(f"Overriding --img-size {args.img_size} to 224 for maxvit_t compatibility.")
        args.img_size = 224
    
    device = get_device(args.device)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True

    train_ds, val_ds = build_datasets(
        args.train_dir,
        args.val_dir,
        args.img_size,
        args.use_dicom,
        args.use_preprocessing,
        args.use_clahe_only,
    )
    
    train_ds, test_ds = split_dataset_stratified(train_ds, test_fraction=args.train_test_split, seed=42)
    output_dir = Path(args.output_dir)
    test_split_path = save_test_split(
        output_dir=output_dir,
        train_dataset=train_ds,
        test_dataset=test_ds,
        train_dir=args.train_dir,
        train_test_split=args.train_test_split,
    )
    
    if args.mini:
        train_ds = make_mini_subset(train_ds, fraction=0.25, seed=42)
        test_ds = make_mini_subset(test_ds, fraction=0.25, seed=43)
        print("Mini mode enabled: using 25% of train and test sets.")
        print(f"Mini train samples: {len(train_ds)}")
        print(f"Mini test samples: {len(test_ds)}")
    
    classes = _dataset_classes(train_ds)
    class_weights = compute_class_weights(train_ds, num_classes=len(classes))

    results = []

    print(f"Classes: {classes}")
    print(f"Class weights: {class_weights.tolist()}")
    print(f"Device: {device}")
    print(f"CUDA devices visible: {torch.cuda.device_count()}")
    print(f"Data parallel enabled: {args.data_parallel}")
    print(f"Using DICOM loader: {args.use_dicom}")
    print(f"Using per-image preprocessing: {args.use_preprocessing}")
    print(f"Using CLAHE with ImageNet normalization: {args.use_clahe_only}")
    print(f"Mini mode: {args.mini}")
    print(f"Train/test split: {args.train_test_split}")
    print(f"Saved test split: {test_split_path}")
    print(f"Total training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"Use LAF (Large Adaptive Filter): {args.use_laf} (kernels={args.laf_kernel_size1},{args.laf_kernel_size2})")
    print(f"Use AlignNorm: {args.use_alignnorm} (temperature={args.alignnorm_temperature}, scale={args.alignnorm_scale})")

    for model_name in args.models:
        batch_size = args.batch_size
        train_loader, val_loader, test_loader = build_loaders(
            train_ds, val_ds, test_ds, batch_size, args.num_workers
        )
        result = train_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_weights=class_weights,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            output_dir=output_dir,
            pretrained=args.pretrained,
            debug=args.debug,
            img_size=args.img_size,            
            use_preprocessing=args.use_preprocessing,
            use_clahe_only=args.use_clahe_only,
            batch_size=batch_size,
            mini=args.mini,
            train_test_split=args.train_test_split,
            test_split_path=str(test_split_path),
            fusion_convnext_model=args.fusion_convnext_model,
            fusion_swin_v2_model=args.fusion_swin_v2_model,
            fusion_freeze_epochs=args.fusion_freeze_epochs,
            fusion_unfreeze_lr_factor=args.fusion_unfreeze_lr_factor,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            data_parallel=args.data_parallel,
            gradient_clip=args.gradient_clip,
            warmup_epochs=args.warmup_epochs,
            transformer_lr=args.transformer_lr,
            use_laf=args.use_laf,
            use_alignnorm=args.use_alignnorm,
            laf_kernel_size1=args.laf_kernel_size1,
            laf_kernel_size2=args.laf_kernel_size2,
            alignnorm_temperature=args.alignnorm_temperature,
            alignnorm_scale=args.alignnorm_scale,
        )
        results.append(result)

    results.sort(key=lambda item: item["best_val_f1"], reverse=True)
    summary_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "img_size": args.img_size,
        "use_preprocessing": args.use_preprocessing,
        "use_clahe_only": args.use_clahe_only,
        "batch_size": args.batch_size,
        "mini": args.mini,
        "data_parallel": args.data_parallel,
        "train_test_split": args.train_test_split,
        "test_split_path": str(test_split_path),
        "total_training_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "fusion_freeze_epochs": args.fusion_freeze_epochs,
        "fusion_unfreeze_lr_factor": args.fusion_unfreeze_lr_factor,
        "fusion_convnext_model": args.fusion_convnext_model,
        "fusion_swin_v2_model": args.fusion_swin_v2_model,
        "gradient_clip": args.gradient_clip,
        "warmup_epochs": args.warmup_epochs,
        "transformer_lr": args.transformer_lr,
        "use_laf": args.use_laf,
        "laf_kernel_size1": args.laf_kernel_size1,
        "laf_kernel_size2": args.laf_kernel_size2,
        "use_alignnorm": args.use_alignnorm,
        "alignnorm_temperature": args.alignnorm_temperature,
        "alignnorm_scale": args.alignnorm_scale,
        "results": results,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\nFinal ranking by best validation F1:")
    for rank, item in enumerate(results, start=1):
        print(f"{rank}. {item['model']}: best_val_f1={item['best_val_f1']:.4f}")
        print(f"   stopped_epoch={item.get('stopped_epoch', 'n/a')}")
        print(f"   checkpoint: {item['best_checkpoint']}")
        test_metrics = item.get("test_metrics") or {}
        if test_metrics:
            print(
                f"   test_f1={test_metrics.get('f1', 0.0):.4f} "
                f"test_acc={test_metrics.get('accuracy', 0.0):.4f} "
                f"test_precision={test_metrics.get('precision', 0.0):.4f} "
                f"test_recall={test_metrics.get('recall', 0.0):.4f} "
                f"test_auroc={test_metrics.get('auroc', 0.0):.4f}"
            )


if __name__ == "__main__":
    main()
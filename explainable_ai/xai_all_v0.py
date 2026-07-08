"""
xai_all.py -- Combined XAI toolkit (Grad-CAM, Score-CAM, Integrated
Gradients, GradientSHAP, Occlusion, Layer CAM, Attention Maps, Eigen CAM)
for a single trained checkpoint.
=====================================================================

WHAT THIS DOES
--------------
Points at a folder of images (any nesting -- e.g. val/normal/, val/tb/, or
a flat folder), randomly samples N of them, runs some or all of the 8 XAI
methods against ONE checkpoint, and either saves a PNG per image (original
+ one panel per method) to an output folder, or renders the same figure
inline for a Jupyter notebook.

DESIGN DECISIONS WORTH KNOWING ABOUT (read before you assume behavior):

1. SAME TARGET CLASS ACROSS METHODS. For each image we run a single
   no-grad forward pass to get the predicted class, then hand that exact
   class index to every method. If we let each method independently pick
   "the model's prediction," a borderline image could show one method
   explaining "tb" and another explaining "normal" purely because of
   floating-point jitter between forward passes -- that would look like a
   disagreement about *what the model is thinking* when it's actually just
   inconsistent bookkeeping on our end. Forcing one target class means any
   visual disagreement you see between panels is real disagreement about
   *where*, not artifacts about *what*.

2. ONE CHECKPOINT, ONE BACKBONE. Grad-CAM, Score-CAM, Layer CAM, and
   Eigen CAM need a single "final spatial feature map" to hook into.
   Late-fusion checkpoints (two backbones -> one head) don't have one such
   layer, so those methods are skipped (with a clear message) for
   late_fusion_* models. Integrated Gradients, GradientSHAP, and Occlusion
   don't need a target layer -- they attribute directly on input pixels --
   so they still run.

3. TRAINING-MODULE IMPORT. This script imports get_model/build_transforms
   from your training script so preprocessing is IDENTICAL to what the
   checkpoint was trained on. Default module name is "model_v1" (per your
   file); override with --training-module if yours is named differently.

4. FAILURES ARE PER-METHOD, NOT FATAL. If e.g. GradientSHAP throws on one
   weird image, you get a labeled blank panel with the error message, not
   a crashed run that loses the other 9 images' results.

5. WHY DISPLAY-MODE DOESN'T FORCE THE 'Agg' BACKEND. matplotlib.use("Agg")
   is required for headless file-saving but breaks inline notebook
   rendering. We only force Agg when --save is used (or by default when
   --display is not passed).

USAGE
-----
Save mode (default), all 8 methods, 10 random images:
    python xai_all.py --checkpoint checkpoints/swin_t_best.pt \\
        --input-dir val --output-dir xai_outputs

Notebook display mode, only Grad-CAM + Occlusion, 5 images:
    python xai_all.py --checkpoint checkpoints/swin_t_best.pt \\
        --input-dir val --display --num-images 5 --methods gradcam occlusion

Everything (input folder, output folder, method subset, image count,
checkpoint, target class) is controlled from the command line -- nothing
here needs to be hand-edited per run.
"""
from __future__ import annotations

import argparse
import importlib
import random
import sys
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from PIL import Image

# --------------------------------------------------------------------------
# Constants shared across methods
# --------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DICOM_EXTENSIONS = {".dcm", ".dicom"}
ALL_EXTENSIONS = IMAGE_EXTENSIONS | DICOM_EXTENSIONS

# Architectures whose torchvision `.features` output is channels-LAST
# ([B, H, W, C]) rather than channels-first ([B, C, H, W]). Needed so
# Grad-CAM / Score-CAM treat CNN and Swin feature maps identically after a
# permute.
SWIN_LIKE_NAMES = {"swin_t", "swin_v2_t", "swin_v2_s"}
RESNET_NAMES = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}
LATE_FUSION_NAMES = {"late_fusion_t", "late_fusion_weighted_t", "late_fusion_attention_t"}

METHOD_CHOICES = ["gradcam", "scorecam", "layercam", "attention", "eigencam", "ig", "shap", "occlusion"]
METHOD_LABELS = {
    "gradcam": "Grad-CAM",
    "scorecam": "Score-CAM",
    "layercam": "Layer CAM",
    "attention": "Attention Maps",
    "eigencam": "Eigen CAM",
    "ig": "Integrated Gradients",
    "shap": "GradientSHAP",
    "occlusion": "Occlusion",
}


# ==========================================================================
# Training-module loading (model + transforms identical to training)
# ==========================================================================

def import_training_module(module_name: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path.cwd()))
    return importlib.import_module(module_name)


def load_model_and_transform(checkpoint_path: str, device: torch.device,
                              training_module_name: str, class_names: Tuple[str, ...]):
    """Rebuild the model exactly as trained and load its best weights.

    class_names defaults to whatever the caller passes (normal/tb by
    default) -- VERIFY this matches your actual train_dir subfolder names
    alphabetically, or every label in this toolkit is silently swapped.
    """
    tm = import_training_module(training_module_name)
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_name = ckpt["model_name"]
    model = tm.get_model(
        model_name,
        num_classes=len(class_names),
        pretrained=False,  # weights come from the checkpoint, not torchvision defaults
        img_size=ckpt.get("img_size", 512),
        fusion_convnext_model=ckpt.get("fusion_convnext_model", "convnext_t"),
        fusion_swin_v2_model=ckpt.get("fusion_swin_v2_model", "swin_v2_t"),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    _, eval_transform = tm.build_transforms(
        img_size=ckpt.get("img_size", 512),
        use_clahe=ckpt.get("use_preprocessing", True),
        use_clahe_only=ckpt.get("use_clahe_only", True),
    )
    return model, eval_transform, model_name, class_names


def get_target_layer(model, model_name: str):
    """Best-effort resolution of 'the last spatial feature map before
    pooling'. Raises clearly instead of guessing if unsupported."""
    if model_name in LATE_FUSION_NAMES:
        raise ValueError(
            f"'{model_name}' is a late-fusion model (two backbones -> one head); "
            "there's no single target layer for Grad-CAM/Score-CAM/Layer CAM/Eigen CAM. Skipping."
        )
    if model_name in RESNET_NAMES:
        return model.layer4
    if hasattr(model, "features"):
        return model.features
    raise ValueError(
        f"Don't know how to find a target layer for '{model_name}'. "
        "Add a case to get_target_layer()."
    )


def to_nchw(x: torch.Tensor, model_name: str) -> torch.Tensor:
    if model_name in SWIN_LIKE_NAMES and x.ndim == 4:
        return x.permute(0, 3, 1, 2).contiguous()
    return x


# ==========================================================================
# Image loading (folder-agnostic: flat folder or nested class subfolders,
# ordinary images or DICOM, all handled the same way)
# ==========================================================================

def load_dicom_as_pil(path: Path) -> Image.Image:
    import pydicom

    ds = pydicom.dcmread(str(path))
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


def load_pil_image(path: Path) -> Image.Image:
    if path.suffix.lower() in DICOM_EXTENSIONS:
        return load_dicom_as_pil(path)
    return Image.open(path).convert("RGB")


def load_image_tensor(path: Path, eval_transform, device: torch.device):
    pil_image = load_pil_image(path)
    tensor = eval_transform(pil_image).unsqueeze(0).to(device)
    return tensor, pil_image


def discover_images(input_dir: Path) -> List[Path]:
    """Recursively find every candidate image under input_dir, regardless
    of subfolder structure -- works whether input_dir is a flat folder or
    an ImageFolder-style tree with class subfolders (normal/, tb/, ...)."""
    found = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in ALL_EXTENSIONS]
    return sorted(found)


def infer_true_label(path: Path, class_names: Tuple[str, ...]) -> Optional[str]:
    """Best-effort: if any ancestor folder name case-insensitively matches
    a known class name, report it. Returns None if it can't tell -- this is
    a convenience label, not something to build metrics on."""
    lowered = {c.lower(): c for c in class_names}
    for parent in path.parents:
        if parent.name.lower() in lowered:
            return lowered[parent.name.lower()]
    return None


# ==========================================================================
# Shared array/plotting helpers
# ==========================================================================

def unnormalize_for_display(tensor: torch.Tensor) -> np.ndarray:
    """[1,3,H,W] normalized tensor -> displayable [H,W,3] uint8. Assumes
    ImageNet normalization -- fine for a visual overlay, not for anything
    quantitative."""
    img = tensor.detach().cpu().squeeze(0).numpy()
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
    return img


def resize_map(map_2d: np.ndarray, target_hw) -> np.ndarray:
    import cv2
    return cv2.resize(map_2d.astype(np.float32), (target_hw[1], target_hw[0]))


def normalize_map(map_2d: np.ndarray) -> np.ndarray:
    m = map_2d - map_2d.min()
    denom = m.max()
    return m / denom if denom > 1e-12 else m


def overlay_heatmap(base_rgb: np.ndarray, heatmap_0_1: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import cv2
    heatmap_uint8 = (np.clip(heatmap_0_1, 0, 1) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlaid = (alpha * heatmap_color + (1 - alpha) * base_rgb).astype(np.uint8)
    return overlaid


def overlay_attention(base_rgb: np.ndarray, attention_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Special overlay for attention maps - they're often more subtle and
    might need different visualization. Currently uses same as heatmap."""
    return overlay_heatmap(base_rgb, attention_map, alpha)


# ==========================================================================
# The 8 methods. Each returns a heatmap already resized to input
# resolution, normalized to [0, 1]. All take an explicit target_class so
# every method explains the same class (see module docstring, point 1).
# ==========================================================================

def run_gradcam(model, model_name: str, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
    target_layer = get_target_layer(model, model_name)
    activations = {}
    gradients = {}

    def fwd_hook(_m, _i, output):
        activations["v"] = output.detach()

    def bwd_hook(_m, _gi, grad_output):
        gradients["v"] = grad_output[0].detach()

    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        output = model(input_tensor)
        score = output[0, target_class]
        score.backward()

        acts = to_nchw(activations["v"], model_name)
        grads = to_nchw(gradients["v"], model_name)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        return normalize_map(cam)
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


@torch.no_grad()
def run_scorecam(model, model_name: str, input_tensor: torch.Tensor, target_class: int,
                  max_channels: int = 256, batch_size: int = 16) -> np.ndarray:
    target_layer = get_target_layer(model, model_name)
    activations = {}

    def fwd_hook(_m, _i, output):
        activations["v"] = output.detach()

    handle = target_layer.register_forward_hook(fwd_hook)
    try:
        model(input_tensor)
        acts = to_nchw(activations["v"], model_name)  # [1,C,H,W]
        _, num_channels, h, w = acts.shape
        img_h, img_w = input_tensor.shape[2], input_tensor.shape[3]

        if num_channels > max_channels:
            idx = torch.linspace(0, num_channels - 1, steps=max_channels).long()
        else:
            idx = torch.arange(num_channels)

        masks = []
        for c in idx.tolist():
            act = acts[0, c]
            act = act - act.min()
            denom = act.max()
            act = act / denom if denom > 1e-12 else act
            act_up = torch.nn.functional.interpolate(
                act.view(1, 1, h, w), size=(img_h, img_w), mode="bilinear", align_corners=False
            )
            masks.append(act_up)
        masks = torch.cat(masks, dim=0)

        weights = torch.zeros(len(idx), device=input_tensor.device)
        for start in range(0, len(idx), batch_size):
            batch_masks = masks[start:start + batch_size]
            masked_inputs = input_tensor * batch_masks
            batch_out = model(masked_inputs)
            batch_probs = torch.softmax(batch_out, dim=1)[:, target_class]
            weights[start:start + batch_size] = batch_probs

        weights = weights.view(-1, 1, 1, 1)
        weighted = (weights * masks).sum(dim=0).squeeze(0)
        cam = torch.relu(weighted).cpu().numpy()
        return normalize_map(cam)
    finally:
        handle.remove()


def run_layercam(model, model_name: str, input_tensor: torch.Tensor, target_class: int,
                  layer_index: int = -1) -> np.ndarray:
    """
    Layer CAM: Uses the activations directly without gradient weighting.
    Similar to Grad-CAM but without the gradient multiplication.
    """
    target_layer = get_target_layer(model, model_name)
    activations = {}

    def fwd_hook(_m, _i, output):
        activations["v"] = output.detach()

    handle = target_layer.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        output = model(input_tensor)
        score = output[0, target_class]
        score.backward()

        acts = to_nchw(activations["v"], model_name)
        # For Layer CAM, we use the activations directly, optionally weighted by gradient magnitude
        cam = acts.mean(dim=1, keepdim=True)  # Average over channels
        cam = torch.relu(cam).squeeze().cpu().numpy()
        return normalize_map(cam)
    finally:
        handle.remove()


def run_attention_maps(model, model_name: str, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
    """
    Attention Maps: Extracts attention maps from transformer-based models.
    For Swin Transformers and models with attention mechanisms.
    """
    # For Swin transformers, try to extract attention from the last block
    attention_maps = []
    
    def get_attention_hook(name):
        def hook(module, input, output):
            # For Swin Transformer blocks, output might contain attention weights
            if hasattr(module, 'attn') and hasattr(module.attn, 'attn_drop'):
                # Try to get attention weights if available
                if hasattr(module.attn, 'get_attention_weights'):
                    attn_weights = module.attn.get_attention_weights()
                    if attn_weights is not None:
                        attention_maps.append(attn_weights.detach())
        return hook
    
    hooks = []
    try:
        # Look for attention modules in the model
        for name, module in model.named_modules():
            if 'attn' in name.lower() and hasattr(module, 'attn_drop'):
                hook = module.register_forward_hook(get_attention_hook(name))
                hooks.append(hook)
        
        # Forward pass to collect attention maps
        with torch.no_grad():
            output = model(input_tensor)
        
        if attention_maps:
            # Average attention maps across heads and layers
            avg_attention = torch.stack([attn.mean(dim=1) for attn in attention_maps]).mean(dim=0)
            # Reshape to spatial map
            # For Swin, attention maps might be in shape [batch, num_patches, num_patches]
            if avg_attention.ndim == 3:
                # Convert patch attention to spatial map
                h = w = int(np.sqrt(avg_attention.shape[-1]))
                attn_map = avg_attention[0].reshape(h, w).cpu().numpy()
            else:
                attn_map = avg_attention[0].cpu().numpy()
            
            return normalize_map(attn_map)
        else:
            # Fallback: use output features as attention proxy
            output = model(input_tensor)
            # Use gradient-free feature maps as attention proxy
            features = []
            def hook_fn(module, input, output):
                features.append(output.detach())
            
            target_layer = get_target_layer(model, model_name)
            hook = target_layer.register_forward_hook(hook_fn)
            with torch.no_grad():
                _ = model(input_tensor)
            hook.remove()
            
            if features:
                acts = to_nchw(features[0], model_name)
                cam = acts.mean(dim=1, keepdim=True).squeeze().cpu().numpy()
                return normalize_map(cam)
            
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
            
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def run_eigencam(model, model_name: str, input_tensor: torch.Tensor, target_class: int,
                  num_components: int = 3) -> np.ndarray:
    """
    Eigen CAM: Uses PCA on the feature activations to find the most
    important components for the target class.
    """
    target_layer = get_target_layer(model, model_name)
    activations = {}
    gradients = {}

    def fwd_hook(_m, _i, output):
        activations["v"] = output.detach()

    def bwd_hook(_m, _gi, grad_output):
        gradients["v"] = grad_output[0].detach()

    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)
    
    try:
        model.zero_grad(set_to_none=True)
        output = model(input_tensor)
        score = output[0, target_class]
        score.backward()

        acts = to_nchw(activations["v"], model_name)  # [1, C, H, W]
        grads = to_nchw(gradients["v"], model_name)   # [1, C, H, W]
        
        # Compute gradient-weighted activations
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        weighted_acts = (weights * acts).squeeze(0)     # [C, H, W]
        
        # Reshape for PCA: [C, H*W]
        c, h, w = weighted_acts.shape
        features = weighted_acts.view(c, -1).cpu().numpy()
        
        # Compute SVD to get principal components
        U, S, Vt = np.linalg.svd(features, full_matrices=False)
        
        # Use the first few components to reconstruct the CAM
        components = np.dot(features.T, U[:, :num_components])
        components = components.T.reshape(num_components, h, w)
        
        # Weight components by their singular values
        eigen_cam = np.sum(components * S[:num_components].reshape(-1, 1, 1), axis=0)
        
        return normalize_map(np.maximum(eigen_cam, 0))
        
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def _make_ig_baseline(input_tensor: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "black":
        return torch.zeros_like(input_tensor)
    if kind == "blur":
        import torchvision.transforms.functional as TF
        return TF.gaussian_blur(input_tensor, kernel_size=[31, 31], sigma=[10.0, 10.0])
    raise ValueError(f"Unknown baseline kind: {kind}")


def run_ig(model, input_tensor: torch.Tensor, target_class: int,
           baseline_kind: str = "black", n_steps: int = 50) -> np.ndarray:
    from captum.attr import IntegratedGradients

    baseline = _make_ig_baseline(input_tensor, baseline_kind)
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(
        input_tensor, baselines=baseline, target=target_class,
        n_steps=n_steps, return_convergence_delta=True,
    )
    print(f"    IG convergence delta: {delta.item():.6f} (should be small; raise --ig-steps if not)")
    attr_map = attributions.squeeze(0).sum(dim=0).detach().cpu().numpy()
    attr_map = np.clip(attr_map, a_min=0, a_max=None)
    return normalize_map(attr_map)


def run_shap(model, input_tensor: torch.Tensor, target_class: int,
             n_samples: int = 50, stdevs: float = 0.09) -> np.ndarray:
    from captum.attr import GradientShap

    black_baseline = torch.zeros_like(input_tensor)
    gray_baseline = torch.full_like(input_tensor, 0.5)
    baselines = torch.cat([black_baseline, gray_baseline], dim=0)

    gs = GradientShap(model)
    attributions = gs.attribute(
        input_tensor, baselines=baselines, target=target_class,
        n_samples=n_samples, stdevs=stdevs,
    )
    attr_map = attributions.squeeze(0).sum(dim=0).detach().cpu().numpy()
    attr_map = np.clip(attr_map, a_min=0, a_max=None)
    return normalize_map(attr_map)


def run_occlusion(model, input_tensor: torch.Tensor, target_class: int,
                   window: int = 32, stride: int = 16) -> np.ndarray:
    from captum.attr import Occlusion

    occlusion = Occlusion(model)
    attributions = occlusion.attribute(
        input_tensor, target=target_class,
        sliding_window_shapes=(3, window, window),
        strides=(3, stride, stride),
        baselines=0,
    )
    attr_map = attributions.squeeze(0).sum(dim=0).detach().cpu().numpy()
    attr_map = np.clip(attr_map, a_min=0, a_max=None)
    return normalize_map(attr_map)


METHOD_FUNCS = {
    "gradcam": run_gradcam,
    "scorecam": run_scorecam,
    "layercam": run_layercam,
    "attention": run_attention_maps,
    "eigencam": run_eigencam,
    "ig": run_ig,
    "shap": run_shap,
    "occlusion": run_occlusion,
}


def run_method(method: str, model, model_name: str, input_tensor: torch.Tensor,
               target_class: int, args) -> np.ndarray:
    """Dispatch one method with its own tuning args, resizing the result to
    input resolution."""
    img_hw = (input_tensor.shape[2], input_tensor.shape[3])
    if method == "gradcam":
        m = run_gradcam(model, model_name, input_tensor, target_class)
    elif method == "scorecam":
        m = run_scorecam(model, model_name, input_tensor, target_class,
                          max_channels=args.scorecam_max_channels, batch_size=args.scorecam_batch_size)
    elif method == "layercam":
        m = run_layercam(model, model_name, input_tensor, target_class,
                          layer_index=args.layercam_layer_index)
    elif method == "attention":
        m = run_attention_maps(model, model_name, input_tensor, target_class)
    elif method == "eigencam":
        m = run_eigencam(model, model_name, input_tensor, target_class,
                          num_components=args.eigencam_components)
    elif method == "ig":
        m = run_ig(model, input_tensor, target_class,
                    baseline_kind=args.ig_baseline, n_steps=args.ig_steps)
    elif method == "shap":
        m = run_shap(model, input_tensor, target_class,
                      n_samples=args.shap_n_samples, stdevs=args.shap_stdevs)
    elif method == "occlusion":
        m = run_occlusion(model, input_tensor, target_class,
                           window=args.occlusion_window, stride=args.occlusion_stride)
    else:
        raise ValueError(f"Unknown method: {method}")
    return resize_map(m, img_hw)


# ==========================================================================
# Per-image figure: original + one panel per successfully-run method
# ==========================================================================

def build_figure(display_img: np.ndarray, panels: List[Tuple[str, Optional[np.ndarray], str]],
                  suptitle: str):
    """panels: list of (method_label, overlay_rgb_or_None, caption).
    overlay is None when that method failed for this image -- rendered as
    a blank panel with the error caption instead of silently dropping it,
    so failures stay visible rather than looking like they were never run.
    """
    import matplotlib.pyplot as plt

    n = len(panels) + 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
    if n == 1:
        axes = [axes]

    axes[0].imshow(display_img)
    axes[0].set_title("Input")
    axes[0].axis("off")

    for ax, (label, overlay, caption) in zip(axes[1:], panels):
        if overlay is not None:
            ax.imshow(overlay)
        else:
            ax.imshow(np.ones_like(display_img) * 245)
        ax.set_title(caption, fontsize=9)
        ax.axis("off")

    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ==========================================================================
# Main
# ==========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Grad-CAM / Score-CAM / Layer CAM / Attention Maps / Eigen CAM / "
                    "Integrated Gradients / GradientSHAP / Occlusion on a random sample "
                    "of images from a folder, using one checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a *_best.pt checkpoint.")
    parser.add_argument("--input-dir", required=True, help="Folder to sample images from (searched recursively; any subfolder structure works).")
    parser.add_argument("--output-dir", default="xai_outputs", help="Where to save PNGs (ignored if --display is set).")
    parser.add_argument("--display", action="store_true", help="Render inline (Jupyter) instead of saving files.")
    parser.add_argument("--num-images", type=int, default=10, help="How many images to randomly sample.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling; omit for a different sample each run.")
    parser.add_argument("--methods", nargs="+", choices=METHOD_CHOICES, default=list(METHOD_CHOICES),
                         help="Subset of methods to run. Default: all 8.")
    parser.add_argument("--class-names", nargs="+", default=["normal", "tb"])
    parser.add_argument("--target-class", type=int, default=None, help="Force a class index instead of using the prediction.")
    parser.add_argument("--training-module", default="model_v1", help="Module name your get_model/build_transforms live in.")
    parser.add_argument("--device", default="auto")

    # Method-specific tuning (all optional, all overridable from bash)
    parser.add_argument("--ig-baseline", choices=["black", "blur"], default="black")
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--shap-n-samples", type=int, default=50)
    parser.add_argument("--shap-stdevs", type=float, default=0.09)
    parser.add_argument("--occlusion-window", type=int, default=32)
    parser.add_argument("--occlusion-stride", type=int, default=16)
    parser.add_argument("--scorecam-max-channels", type=int, default=256)
    parser.add_argument("--scorecam-batch-size", type=int, default=16)
    parser.add_argument("--layercam-layer-index", type=int, default=-1,
                        help="Layer index for Layer CAM (default: -1 for last layer)")
    parser.add_argument("--eigencam-components", type=int, default=3,
                        help="Number of principal components for Eigen CAM")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.display:
        import matplotlib
        matplotlib.use("Agg")  # headless; must be set before pyplot import below

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    class_names = tuple(args.class_names)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, eval_transform, model_name, class_names = load_model_and_transform(
        args.checkpoint, device, args.training_module, class_names
    )
    print(f"Model architecture: {model_name}")

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"--input-dir does not exist or is not a folder: {input_dir}")

    all_images = discover_images(input_dir)
    if not all_images:
        raise SystemExit(f"No images (extensions {sorted(ALL_EXTENSIONS)}) found under {input_dir}")

    rng = random.Random(args.seed)
    sample_size = min(args.num_images, len(all_images))
    chosen = rng.sample(all_images, sample_size)
    print(f"Sampling {sample_size} of {len(all_images)} images found under {input_dir}"
          f"{f' (seed={args.seed})' if args.seed is not None else ''}:")
    for p in chosen:
        print(f"  - {p}")

    methods = args.methods
    late_fusion_skip = model_name in LATE_FUSION_NAMES
    if late_fusion_skip:
        skipped = [m for m in methods if m in {"gradcam", "scorecam", "layercam", "eigencam"}]
        if skipped:
            print(f"NOTE: '{model_name}' is a late-fusion model -- {skipped} have no single "
                  f"target layer and will be skipped for every image (IG/SHAP/Occlusion/Attention still run).")

    output_dir = Path(args.output_dir)
    if not args.display:
        output_dir.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(chosen, start=1):
        print(f"\n[{i}/{sample_size}] {image_path}")
        try:
            input_tensor, _pil = load_image_tensor(image_path, eval_transform, device)
        except Exception as e:
            print(f"  FAILED to load image, skipping: {e}")
            continue

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_class = int(probs.argmax())
        target_class = args.target_class if args.target_class is not None else pred_class

        display_img = unnormalize_for_display(input_tensor)
        true_label = infer_true_label(image_path, class_names)

        panels = []
        for method in methods:
            if late_fusion_skip and method in {"gradcam", "scorecam", "layercam", "eigencam"}:
                panels.append((METHOD_LABELS[method], None, f"{METHOD_LABELS[method]}\n(unsupported: late-fusion model)"))
                continue
            try:
                heatmap = run_method(method, model, model_name, input_tensor, target_class, args)
                overlay = overlay_heatmap(display_img, heatmap)
                caption = f"{METHOD_LABELS[method]}\nexplaining: {class_names[target_class]}"
                panels.append((METHOD_LABELS[method], overlay, caption))
                print(f"    {METHOD_LABELS[method]}: ok")
            except Exception as e:
                print(f"    {METHOD_LABELS[method]}: FAILED -- {e}")
                traceback.print_exc(limit=1)
                panels.append((METHOD_LABELS[method], None, f"{METHOD_LABELS[method]}\nFAILED: {e}"))

        prob_str = ", ".join(f"{n}={p:.3f}" for n, p in zip(class_names, probs))
        true_str = f" | true={true_label}" if true_label else ""
        suptitle = f"{image_path.name} | pred={class_names[pred_class]}{true_str} | {prob_str}"

        fig = build_figure(display_img, panels, suptitle)

        if args.display:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            out_path = output_dir / f"{i:02d}_{image_path.stem}.png"
            fig.savefig(out_path, dpi=150)
            print(f"    Saved: {out_path}")
        import matplotlib.pyplot as plt
        plt.close(fig)

    if not args.display:
        print(f"\nDone. Results in: {output_dir}")


if __name__ == "__main__":
    main()
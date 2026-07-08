# input a random image and get inference
# python predict.py --image path/to/chest_xray.jpg --model checkpoints/swin_t_best.pt

#!/usr/bin/env python3
"""
TB Classification Prediction Script
Usage:
    python predict.py --image path/to/chest_xray.jpg --model checkpoints/swin_t_best.pt
    python predict.py --image path/to/image.png --model checkpoints/resnet50_best.pt --use-clahe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Import necessary classes from your training script
# If predict.py is in the same directory as your training script, you can import directly:
# from your_training_script import get_model, CLAHEBlurTransform, PerImageStandardize

# Otherwise, copy the required classes here:

class CLAHEBlurTransform:
    """Apply CLAHE and Gaussian blur preprocessing."""
    def __init__(self, blur_radius: float = 0.5, clip_limit: float = 2.0, tile_grid_size: int = 8):
        self.blur_radius = blur_radius
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
        if self.blur_radius and self.blur_radius > 0:
            arr = cv2.GaussianBlur(arr, ksize=(0, 0), sigmaX=self.blur_radius)
        return Image.fromarray(arr, mode="L").convert("RGB")


class PerImageStandardize:
    """Standardize each image to zero mean and unit variance."""
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        std = tensor.std()
        return (tensor - mean) / (std + 1e-8)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """Load model and metadata from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_name = checkpoint.get("model_name", "unknown")
    img_size = checkpoint.get("img_size", 512)
    use_preprocessing = checkpoint.get("use_preprocessing", False)
    use_clahe_only = checkpoint.get("use_clahe_only", True)
    
    print(f"Loading model: {model_name}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Preprocessing: per_image={use_preprocessing}, clahe_only={use_clahe_only}")
    
    # Import and create model
    from model_v1 import get_model  # Adjust import based on your file name
    
    model = get_model(
        model_name,
        num_classes=2,
        pretrained=False,  # We're loading weights from checkpoint
        img_size=img_size,
    )
    
    # Load state dict
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        # Try loading the checkpoint directly as state dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Store metadata for preprocessing
    metadata = {
        "model_name": model_name,
        "img_size": img_size,
        "use_preprocessing": use_preprocessing,
        "use_clahe_only": use_clahe_only,
        "best_val_f1": checkpoint.get("best_val_f1", "unknown"),
        "best_val_epoch": checkpoint.get("best_val_epoch", "unknown"),
    }
    
    return model, metadata


def get_transform(img_size: int, use_preprocessing: bool, use_clahe_only: bool) -> transforms.Compose:
    """Create the appropriate transform based on training configuration."""
    transform_list = []
    
    if use_preprocessing or use_clahe_only:
        transform_list.append(CLAHEBlurTransform())
    
    transform_list.append(transforms.Resize((img_size, img_size)))
    transform_list.append(transforms.ToTensor())
    
    if use_preprocessing and not use_clahe_only:
        # Per-image standardization (as used in training)
        transform_list.append(PerImageStandardize())
    else:
        # ImageNet normalization (compatible with pretrained models)
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    
    return transforms.Compose(transform_list)


def predict_image(
    image_path: str,
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    class_names: list = None,
) -> Dict:
    """
    Predict whether a chest X-ray shows TB or is normal.
    
    Args:
        image_path: Path to the image file
        model: Loaded PyTorch model
        transform: Image preprocessing transform
        device: Torch device
        class_names: List of class names (default: ['normal', 'tb'])
    
    Returns:
        Dictionary with prediction results
    """
    if class_names is None:
        class_names = ['Normal', 'TB']
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open image {image_path}: {e}")
    
    # Apply transforms
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()
    
    # Get probabilities
    prob_normal = probabilities[0][0].item()
    prob_tb = probabilities[0][1].item()
    
    # Calculate confidence
    confidence = max(prob_normal, prob_tb)
    
    result = {
        "image_path": str(image_path),
        "predicted_class": class_names[predicted_class],
        "class_index": predicted_class,
        "probability_normal": prob_normal,
        "probability_tb": prob_tb,
        "confidence": confidence,
        "probabilities": {
            class_names[0]: f"{prob_normal:.4f} ({prob_normal*100:.2f}%)",
            class_names[1]: f"{prob_tb:.4f} ({prob_tb*100:.2f}%)",
        }
    }
    
    return result


def predict_batch(
    image_paths: list,
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    class_names: list = None,
) -> list:
    """Predict multiple images."""
    results = []
    for image_path in image_paths:
        try:
            result = predict_image(image_path, model, transform, device, class_names)
            results.append(result)
        except Exception as e:
            results.append({
                "image_path": str(image_path),
                "error": str(e)
            })
    return results


def format_prediction(result: Dict) -> str:
    """Format prediction result for display."""
    if "error" in result:
        return f"Error processing {result['image_path']}: {result['error']}"
    
    pred_class = result["predicted_class"]
    confidence = result["confidence"]
    
    # Visual indicator
    if pred_class == "TB":
        recommendation = "Further clinical evaluation recommended."
    else:
        recommendation = "No signs of TB detected."
    
    output = f"""
{icon} PREDICTION RESULT
{'='*50}
Image: {result['image_path']}
Prediction: {pred_class.upper()}
Confidence: {confidence:.2%}

Probability Breakdown:
  • Normal: {result['probabilities']['Normal']}
  • TB:     {result['probabilities']['TB']}

{recommendation}
{'='*50}
"""
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Predict TB from chest X-ray images using trained model."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to chest X-ray image file (JPG, PNG, or DICOM)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save prediction results to JSON file."
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process multiple images (--image can be a directory or multiple files)."
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    model, metadata = load_model_from_checkpoint(args.model, device)
    
    # Get transform based on training configuration
    transform = get_transform(
        metadata["img_size"],
        metadata["use_preprocessing"],
        metadata["use_clahe_only"]
    )
    
    # Process image(s)
    if args.batch or Path(args.image).is_dir():
        if Path(args.image).is_dir():
            image_dir = Path(args.image)
            image_paths = sorted([
                str(p) for p in image_dir.glob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".dcm"}
            ])
        else:
            # Assume comma-separated list
            image_paths = [p.strip() for p in args.image.split(",")]
        
        if not image_paths:
            print("No images found!")
            return
        
        results = predict_batch(image_paths, model, transform, device)
        
        for result in results:
            print(format_prediction(result))
        
    else:
        # Single image prediction
        result = predict_image(args.image, model, transform, device)
        print(format_prediction(result))
        
        results = [result]
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time

# Enable GPU memory growth and optimizations
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)]
        )
    except RuntimeError as e:
        print(e)

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

# =====================================================
# CONFIG
# =====================================================

BACKBONE = "efficientnetb0"
IMG_SIZE = (256, 256)
BATCH_SIZE = 128  # H100 can handle large batches
MODEL_WEIGHTS = "best_lung_model.h5"
INPUT_FOLDERS = ["train", "val"]
OUTPUT_SUFFIX = "-segmented"
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
GAMMA = 0.8

# Soft masking settings
DIM_FACTOR = 0.3  # How much to dim non-lung areas (0.0 = black, 1.0 = no dimming)
BBOX_PADDING = 10  # Padding around each lung bounding box in pixels
SOFT_MASK_BLUR = 15  # Gaussian blur kernel size for smoothing bounding box edges

# TensorRT settings
USE_TENSORRT = False  # Set to True only if you have TensorRT installed
TENSORRT_MODEL_PATH = "lung_model_trt"

# =====================================================
# OPTIMIZED PREPROCESSOR
# =====================================================

class UltraFastPreprocessor:
    def __init__(self, clahe_clip=2.0, tile_size=(8,8), gamma=0.8):
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=tile_size)
        self.gamma_table = np.array(
            [((i / 255.0) ** (1.0/gamma)) * 255 for i in np.arange(256)],
            dtype=np.uint8
        )
    
    def __call__(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = self.clahe.apply(gray)
        gray = cv2.LUT(gray, self.gamma_table)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading model...")
preprocess_input = sm.get_preprocessing(BACKBONE)

model = sm.Unet(
    BACKBONE,
    encoder_weights=None,
    classes=1,
    activation="sigmoid",
    decoder_block_type="upsampling"
)
model.load_weights(MODEL_WEIGHTS)

# Compile with optimizations
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    run_eagerly=False
)

print("Model loaded successfully!")

# =====================================================
# TENSORRT CONVERSION (OPTIONAL)
# =====================================================

if USE_TENSORRT:
    try:
        import tensorrt as trt
        print("TensorRT found, attempting conversion...")
        
        # Create a wrapper function for the model
        @tf.function
        def model_predict(x):
            return model(x, training=False)
        
        # Save with signatures
        tf.saved_model.save(
            model,
            "temp_saved_model",
            signatures={
                'serving_default': model_predict.get_concrete_function(
                    tf.TensorSpec(shape=[None, 256, 256, 3], dtype=tf.float32)
                )
            }
        )
        print("SavedModel created successfully!")
        
        # Convert to TensorRT
        from tensorflow.python.compiler.tensorrt import trt_convert as trt_conv
        
        print("Converting to TensorRT...")
        converter = trt_conv.TrtGraphConverterV2(
            input_saved_model_dir="temp_saved_model",
            precision_mode=trt_conv.TrtPrecisionMode.FP16,
            maximum_cached_engines=100,
            use_dynamic_shape=True,
            dynamic_shape_profile_strategy=trt_conv.ProfileStrategy.RANGE
        )
        
        converter.convert()
        converter.save(TENSORRT_MODEL_PATH)
        print(f"TensorRT model saved to: {TENSORRT_MODEL_PATH}")
        
        # Load TensorRT model
        model = tf.saved_model.load(TENSORRT_MODEL_PATH)
        print("Using TensorRT model!")
        
    except Exception as e:
        print(f"TensorRT conversion failed: {e}")
        print("Falling back to regular model")
        USE_TENSORRT = False

# =====================================================
# BATCH PROCESSING
# =====================================================

def process_batch(image_paths, save_paths, model, preprocessor, use_trt=False):
    """Process images in batch with bounding-box soft masking"""
    
    if not image_paths:
        return
    
    batch_imgs = []
    batch_original = []
    batch_info = []
    
    # Read and preprocess all images
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed: {img_path}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        
        # Preprocess
        proc = preprocessor(img)
        proc = cv2.resize(proc, IMG_SIZE)
        
        batch_imgs.append(proc)
        batch_original.append(img)
        batch_info.append((orig_h, orig_w))
    
    if not batch_imgs:
        return
    
    # Convert to numpy array
    batch_imgs = np.array(batch_imgs, dtype=np.float32)
    
    # Preprocess for model (if not using TensorRT)
    if not use_trt:
        batch_imgs = preprocess_input(batch_imgs)
    
    # GPU prediction
    if use_trt:
        batch_pred = model(batch_imgs).numpy()
    else:
        batch_pred = model.predict(
            batch_imgs, 
            verbose=0, 
            batch_size=min(BATCH_SIZE, len(batch_imgs))
        )
    
    # Process results
    for pred, img_rgb, info, save_path in zip(batch_pred, batch_original, batch_info, save_paths):
        orig_h, orig_w = info
        
        # Get soft mask (probability map)
        soft_mask = cv2.resize(
            pred.squeeze(), 
            (orig_w, orig_h), 
            interpolation=cv2.INTER_LINEAR
        )
        soft_mask = np.clip(soft_mask, 0.0, 1.0)
        
        # -------------------------------------
        # FIND LUNGS USING BOUNDING BOXES (Program 1 logic)
        # -------------------------------------
        
        # Create binary mask for connected components
        binary_mask_for_components = (soft_mask > 0.5).astype(np.uint8)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask_for_components,
            connectivity=8
        )
        
        components = []
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < 100:
                continue
            components.append((label_id, area))
        
        components.sort(key=lambda x: x[1], reverse=True)
        components = components[:2]
        
        # -------------------------------------
        # IF NO LUNG FOUND
        # -------------------------------------
        
        if len(components) == 0:
            # No lungs detected - create fully dimmed image
            output_rgb = (img_rgb.astype(np.float32) * DIM_FACTOR).astype(np.uint8)
        else:
            # -------------------------------------
            # CREATE BOUNDING BOX MASK
            # -------------------------------------
            
            # Start with a mask of DIM_FACTOR everywhere (dimmed background)
            bbox_mask = np.full((orig_h, orig_w), DIM_FACTOR, dtype=np.float32)
            
            # Set bounding box regions to 1.0 (full original intensity)
            for label_id, _ in components:
                x = stats[label_id, cv2.CC_STAT_LEFT]
                y = stats[label_id, cv2.CC_STAT_TOP]
                w = stats[label_id, cv2.CC_STAT_WIDTH]
                h = stats[label_id, cv2.CC_STAT_HEIGHT]
                
                # Apply padding
                x_pad = max(0, x - BBOX_PADDING)
                y_pad = max(0, y - BBOX_PADDING)
                w_pad = min(orig_w - x_pad, w + 2 * BBOX_PADDING)
                h_pad = min(orig_h - y_pad, h + 2 * BBOX_PADDING)
                
                bbox_mask[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad] = 1.0
            
            # Apply Gaussian blur to smooth the bounding box edges
            blur_kernel = SOFT_MASK_BLUR if SOFT_MASK_BLUR % 2 == 1 else SOFT_MASK_BLUR + 1
            bbox_mask_smooth = cv2.GaussianBlur(bbox_mask, (blur_kernel, blur_kernel), 0)
            bbox_mask_smooth = np.clip(bbox_mask_smooth, DIM_FACTOR, 1.0)
            
            # -------------------------------------
            # APPLY SOFT MASK: INSIDE BOXES = ORIGINAL, OUTSIDE = DIMMED
            # -------------------------------------
            
            # Expand intensity map to 3 channels for RGB
            intensity_3ch = np.stack([bbox_mask_smooth, bbox_mask_smooth, bbox_mask_smooth], axis=2)
            
            # Apply intensity scaling to the original image
            output_rgb = (img_rgb.astype(np.float32) * intensity_3ch).astype(np.uint8)
        
        # Convert and save
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

# =====================================================
# MAIN PROCESSING
# =====================================================

print("\n" + "="*60)
print("H100 GPU OPTIMIZED: BOUNDING BOX SOFT MASKING")
print("="*60)
print(f"TensorRT: {'Enabled' if USE_TENSORRT else 'Disabled'}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Dim Factor: {DIM_FACTOR} (lower = more dimming)")
print(f"BBox Padding: {BBOX_PADDING}px around each lung")
print(f"Soft Mask Blur: {SOFT_MASK_BLUR}px")
print(f"Mode: Inside bounding boxes = original, outside = dimmed")
print("="*60 + "\n")

preprocessor = UltraFastPreprocessor()
total_start = time.time()

for dataset in INPUT_FOLDERS:
    output_dataset = dataset + OUTPUT_SUFFIX
    
    for cls in ["normal", "tb"]:
        input_dir = os.path.join(dataset, cls)
        output_dir = os.path.join(output_dataset, cls)
        os.makedirs(output_dir, exist_ok=True)
        
        images = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        
        if not images:
            print(f"No images found in {input_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {dataset}/{cls}")
        print(f"Total images: {len(images)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        processed_count = 0
        
        # Process in batches with progress bar
        with tqdm(total=len(images), desc=f"{cls}", unit="img") as pbar:
            for i in range(0, len(images), BATCH_SIZE):
                batch_images = images[i:i+BATCH_SIZE]
                
                image_paths = [os.path.join(input_dir, img) for img in batch_images]
                save_paths = [os.path.join(output_dir, img) for img in batch_images]
                
                # Process batch
                process_batch(
                    image_paths, 
                    save_paths, 
                    model, 
                    preprocessor, 
                    use_trt=USE_TENSORRT
                )
                
                processed_count += len(batch_images)
                pbar.update(len(batch_images))
        
        elapsed = time.time() - start_time
        speed = len(images) / elapsed if elapsed > 0 else 0
        
        print(f"\nFinished {dataset}/{cls}")
        print(f"Time: {elapsed:.2f} seconds")
        print(f"Speed: {speed:.2f} images/second")
        print(f"{'='*60}\n")

total_elapsed = time.time() - total_start
print(f"\n{'='*60}")
print("PROCESSING COMPLETE!")
print(f"Total time: {total_elapsed:.2f} seconds")
print(f"Output folders: train{OUTPUT_SUFFIX}/, val{OUTPUT_SUFFIX}/")
print(f"{'='*60}")

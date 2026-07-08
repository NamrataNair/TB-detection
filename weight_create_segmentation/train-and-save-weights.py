import os
import cv2
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import albumentations as A

# Set environment variable for segmentation_models framework
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
BACKBONE = 'efficientnetb0'
preprocess_input = sm.get_preprocessing(BACKBONE)
# ---------------------------------------------------------
# 1. HYPERPARAMETERS & SETUP
# ---------------------------------------------------------
# FIX: Switched from efficientnetb0 to resnet34.
#      The `efficientnet` pip package fetches weights from a dead GitHub URL
#      (Callidior/keras-applications 404). resnet34 weights load cleanly via
#      segmentation_models and do not hit this issue.

BATCH_SIZE  = 8
EPOCHS      = 70
LR          = 0.0002
IMG_SIZE    = (256, 256)
THRESHOLD   = 0.5
LR_REDUCE_FACTOR   = 0.5
LR_PATIENCE        = 10
EARLY_STOP_PATIENCE = 20
SEED = 42

# Set all random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Kaggle Paths
IMG_DIR    = 'CXR_png'
MASK_DIR   = 'masks'
OUTPUT_DIR = 'weights'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 2. PRE-FILTER VALID IMAGE-MASK PAIRS
# ---------------------------------------------------------
all_images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])

valid_pairs = [
    img_name for img_name in all_images
    if os.path.exists(os.path.join(MASK_DIR, img_name.replace('.png', '_mask.png')))
]

skipped = len(all_images) - len(valid_pairs)
if skipped > 0:
    print(f"Warning: Skipped {skipped} images with missing masks.")

train_images, val_images = train_test_split(valid_pairs, test_size=0.2, random_state=SEED)
print(f"Dataset Split: {len(train_images)} training | {len(val_images)} validation images.")

# ---------------------------------------------------------
# 3. AUGMENTATION PIPELINE (training only)
# ---------------------------------------------------------
# FIX 1: Replaced deprecated ShiftScaleRotate with A.Affine (current API).
# FIX 2: Removed var_limit from GaussNoise — argument removed in newer albumentations.
#         Use noise_scale_factor or just rely on defaults instead.
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
    A.Affine(
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        scale=(0.9, 1.1),
        rotate=(-10, 10),
        p=0.5
    ),
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    A.GaussNoise(p=0.2),   # FIX: no var_limit; use albumentations defaults
])

# ---------------------------------------------------------
# 4. DATA GENERATOR
# ---------------------------------------------------------
def data_generator(image_names, image_dir, mask_dir, batch_size, augment=False):
    """
    Infinite generator yielding (images, masks) batches.
    - replace=False: no duplicate images within one batch.
    - BGR → RGB: correct color order for ImageNet-pretrained encoder.
    - Augmentation applied jointly to image + mask (keeps alignment).
    - Batch size always guaranteed (pairs pre-validated, no skips inside loop).
    """
    while True:
        batch_names   = np.random.choice(a=image_names, size=batch_size, replace=False)
        batch_images  = []
        batch_masks   = []

        for img_name in batch_names:
            mask_name = img_name.replace('.png', '_mask.png')
            img_path  = os.path.join(image_dir, img_name)
            mask_path = os.path.join(mask_dir,  mask_name)

            img = cv2.imread(img_path)
            if img is None:
                continue
            img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # FIX: BGR → RGB
            img  = cv2.resize(img, IMG_SIZE)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            mask = cv2.resize(mask, IMG_SIZE)

            if augment:
                augmented = train_aug(image=img, mask=mask)
                img  = augmented['image']
                mask = augmented['mask']

            # FIX: explicit float32 cast — numpy defaults to float64 after /255.0
            # but TF/segmentation_models metrics require float32, causing:
            # TypeError: Input 'y' of 'Mul' Op has type float32 that does not
            # match type float64 of argument 'x'
            img = preprocess_input(img.astype(np.float32))
            mask = (mask / 255.0).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)   # Shape: (H, W, 1)

            batch_images.append(img)
            batch_masks.append(mask)

        yield (np.array(batch_images, dtype=np.float32),
               np.array(batch_masks,  dtype=np.float32))

# ---------------------------------------------------------
# 5. BUILD THE U-NET MODEL
# ---------------------------------------------------------
print("Building U-Net model with efficientnet0 backbone...")
model = sm.Unet(
    BACKBONE,
    encoder_weights=None,
    classes=1,
    activation='sigmoid',
    decoder_block_type='upsampling'
)

# Combined Dice + BCE loss: more stable than Dice alone at initialization.
# FIX: Must use sm.losses.BinaryCELoss() instead of tf.keras.losses.BinaryCrossentropy()
#      because segmentation_models' __add__ only accepts objects inheriting from sm Loss.
dice_loss  = sm.losses.DiceLoss()
bce_loss   = sm.losses.BinaryCELoss()
total_loss = dice_loss + bce_loss

metrics = [
    sm.metrics.IOUScore(threshold=THRESHOLD),
    sm.metrics.FScore(threshold=THRESHOLD)
]

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
model.compile(optimizer=optimizer, loss=total_loss, metrics=metrics)
model.summary()

# ---------------------------------------------------------
# 6. TRAINING CALLBACKS
# ---------------------------------------------------------
model_save_path = os.path.join(OUTPUT_DIR, 'best_lung_model.h5')

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=LR_REDUCE_FACTOR,
    patience=LR_PATIENCE,
    min_lr=1e-6,
    verbose=1
)

# ModelCheckpoint owns the best-weights responsibility.
# EarlyStopping has restore_best_weights=False to avoid conflict.
checkpoint = ModelCheckpoint(
    model_save_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOP_PATIENCE,
    restore_best_weights=False
)

# ---------------------------------------------------------
# 7. EXECUTE TRAINING
# ---------------------------------------------------------
print("Starting training...")

# max(1, ...) guards against val_steps = 0 when val set < batch_size
train_steps = max(1, len(train_images) // BATCH_SIZE)
val_steps   = max(1, len(val_images)   // BATCH_SIZE)

history = model.fit(
    data_generator(train_images, IMG_DIR, MASK_DIR, BATCH_SIZE, augment=True),
    steps_per_epoch=train_steps,
    validation_data=data_generator(val_images, IMG_DIR, MASK_DIR, BATCH_SIZE, augment=False),
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=[reduce_lr, checkpoint, early_stop]
)

# ---------------------------------------------------------
# 8. PLOT AND SAVE TRAINING HISTORY
# ---------------------------------------------------------
def plot_history(history, output_dir):
    """Save loss and IOU training curves for overfitting diagnosis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'],     label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss',   linewidth=2, linestyle='--')
    axes[0].set_title('Loss Curve', fontsize=14)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['iou_score'],     label='Train IOU', linewidth=2)
    axes[1].plot(history.history['val_iou_score'], label='Val IOU',   linewidth=2, linestyle='--')
    axes[1].set_title('IOU Score Curve', fontsize=14)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('IOU Score')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {save_path}")

plot_history(history, OUTPUT_DIR)

# ---------------------------------------------------------
# 9. VISUALIZE AND SAVE PREDICTIONS
# ---------------------------------------------------------
def view_and_save_predictions(model, image_list, img_dir, mask_dir, output_dir, num_images=5):
    """
    Picks random validation images, runs inference, saves side-by-side comparisons.
    BGR → RGB applied consistently with the training pipeline.
    """
    print(f"\nGenerating {num_images} prediction visualizations...")
    random_samples = random.sample(image_list, min(num_images, len(image_list)))

    for i, img_name in enumerate(random_samples):
        img_path  = os.path.join(img_dir, img_name)
        mask_name = img_name.replace('.png', '_mask.png')
        mask_path = os.path.join(mask_dir, mask_name)

        orig_img = cv2.imread(img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_img = cv2.resize(orig_img, IMG_SIZE)
        img_input = np.expand_dims(
            preprocess_input(orig_img.astype(np.float32)),
            axis=0
        )

        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if true_mask is not None:
            true_mask = cv2.resize(true_mask, IMG_SIZE)
        else:
            true_mask = np.zeros(IMG_SIZE, dtype=np.uint8)

        pred_mask = model.predict(img_input, verbose=0)[0]
        pred_mask = (pred_mask > THRESHOLD).astype(np.uint8) * 255

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Sample {i+1}: {img_name}", fontsize=12)

        axes[0].imshow(orig_img);                        axes[0].set_title("Original X-Ray");    axes[0].axis('off')
        axes[1].imshow(true_mask, cmap='gray');          axes[1].set_title("Ground Truth Mask"); axes[1].axis('off')
        axes[2].imshow(pred_mask[:, :, 0], cmap='gray'); axes[2].set_title("Model Prediction");  axes[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'prediction_sample_{i+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

print(f"\nLoading best weights from: {model_save_path}")
model.load_weights(model_save_path)
view_and_save_predictions(model, val_images, IMG_DIR, MASK_DIR, OUTPUT_DIR, num_images=5)

print("\nAll done! Outputs saved to:", OUTPUT_DIR)
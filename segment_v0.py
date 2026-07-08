import os
import cv2
import numpy as np
import tensorflow as tf

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

# =====================================================
# CONFIG
# =====================================================

BACKBONE = "efficientnetb0"
IMG_SIZE = (256, 256)

MODEL_WEIGHTS = "best_lung_model.h5"

INPUT_FOLDERS = ["train", "val"]

OUTPUT_SUFFIX = "-segmented"

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

GAMMA = 0.8

# =====================================================
# GAMMA
# =====================================================

def apply_gamma(image, gamma=1.0):

    inv_gamma = 1.0 / gamma

    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255
         for i in np.arange(256)],
        dtype=np.uint8
    )

    return cv2.LUT(image, table)

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading lung segmentation model...")

preprocess_input = sm.get_preprocessing(BACKBONE)

model = sm.Unet(
    BACKBONE,
    encoder_weights=None,
    classes=1,
    activation="sigmoid",
    decoder_block_type="upsampling"
)

model.load_weights(MODEL_WEIGHTS)

print("Model loaded.")

# =====================================================
# PROCESS ONE IMAGE
# =====================================================

def process_image(image_path, save_path):

    img = cv2.imread(image_path)

    if img is None:
        print("Failed:", image_path)
        return

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    orig_h, orig_w = img.shape[:2]

    # -------------------------------------
    # PREPROCESS FOR SEGMENTATION
    # -------------------------------------

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )

    gray = clahe.apply(gray)

    gray = apply_gamma(
        gray,
        GAMMA
    )

    proc = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2RGB
    )

    proc = cv2.resize(
        proc,
        IMG_SIZE
    )

    inp = preprocess_input(
        proc.astype(np.float32)
    )

    inp = np.expand_dims(
        inp,
        axis=0
    )

    # -------------------------------------
    # SEGMENT
    # -------------------------------------

    pred = model.predict(
        inp,
        verbose=0
    )[0]

    mask = (
        pred > 0.5
    ).astype(np.uint8)

    mask = cv2.resize(
        mask.squeeze(),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    )

    mask = (
        mask * 255
    ).astype(np.uint8)

    # -------------------------------------
    # FIND LUNGS
    # -------------------------------------

    num_labels, labels, stats, _ = \
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )

    components = []

    for label_id in range(1, num_labels):

        area = stats[
            label_id,
            cv2.CC_STAT_AREA
        ]

        if area < 100:
            continue

        components.append(
            (label_id, area)
        )

    components.sort(
        key=lambda x: x[1],
        reverse=True
    )

    components = components[:2]

    # -------------------------------------
    # IF NO LUNG FOUND
    # -------------------------------------

    if len(components) == 0:

        print("No lungs detected:", image_path)

        output = np.zeros_like(img)

    else:

        # ---------------------------------
        # KEEP ONLY INSIDE BOXES
        # ---------------------------------

        output = np.zeros_like(img)

        for label_id, _ in components:

            x = stats[
                label_id,
                cv2.CC_STAT_LEFT
            ]

            y = stats[
                label_id,
                cv2.CC_STAT_TOP
            ]

            w = stats[
                label_id,
                cv2.CC_STAT_WIDTH
            ]

            h = stats[
                label_id,
                cv2.CC_STAT_HEIGHT
            ]

            output[
                y:y+h,
                x:x+w
            ] = img[
                y:y+h,
                x:x+w
            ]

    # -------------------------------------
    # SAVE
    # -------------------------------------

    output = cv2.cvtColor(
        output,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        save_path,
        output
    )

# =====================================================
# PROCESS DATASET
# =====================================================

for dataset in INPUT_FOLDERS:

    output_dataset = dataset + OUTPUT_SUFFIX

    for cls in ["normal", "tb"]:

        input_dir = os.path.join(
            dataset,
            cls
        )

        output_dir = os.path.join(
            output_dataset,
            cls
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        images = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        print(
            f"\nProcessing {dataset}/{cls}"
        )

        for i, image_name in enumerate(images):

            src = os.path.join(
                input_dir,
                image_name
            )

            dst = os.path.join(
                output_dir,
                image_name
            )

            process_image(
                src,
                dst
            )

            if (i + 1) % 100 == 0:

                print(
                    f"{i+1}/{len(images)}"
                )

# =====================================================
# DONE
# =====================================================

print("\nDONE")
print("Generated folders:")
print("train-lb/")
print("val-lb/")

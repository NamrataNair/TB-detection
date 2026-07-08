from pathlib import Path

import cv2
import numpy as np


# =====================================================
# CONFIGURATION
# =====================================================

INPUT_DIR = Path("val") # change the name here to the folder with inversion problem
OUTPUT_DIR = Path("val-inverted") # name your output folder 

VALID_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# =====================================================
# INVERSION DECISION
# =====================================================

def should_invert_xray(
    image: np.ndarray,
    patch_fraction: float = 0.05,
    threshold: int = 127,
):
    """
    Returns True if image should be inverted.

    Rule:
        3+ white corners -> invert
        3+ black corners -> leave unchanged
    """

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    ph = max(10, int(h * patch_fraction))
    pw = max(10, int(w * patch_fraction))

    patches = [
        gray[0:ph, 0:pw],          # top-left
        gray[0:ph, w-pw:w],        # top-right
        gray[h-ph:h, 0:pw],        # bottom-left
        gray[h-ph:h, w-pw:w],      # bottom-right
    ]

    # Median is more robust than mean
    corner_values = [float(np.median(p)) for p in patches]

    white_count = sum(v > threshold for v in corner_values)
    black_count = 4 - white_count

    invert_needed = white_count >= 2 # mention the number of coreners to check if inversion is needed.

    return (
        invert_needed,
        corner_values,
        white_count,
        black_count,
    )


# =====================================================
# PROCESS DATASET
# =====================================================

def process_dataset():

    OUTPUT_DIR.mkdir(exist_ok=True)

    total_images = 0
    inverted_images = 0

    for image_path in INPUT_DIR.rglob("*"):

        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        rel_path = image_path.relative_to(INPUT_DIR)
        output_path = OUTPUT_DIR / rel_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            print(f"Could not read: {image_path}")
            continue

        (
            invert_needed,
            corner_values,
            white_count,
            black_count,
        ) = should_invert_xray(image)

        if invert_needed:
            image = 255 - image
            inverted_images += 1
            action = "INVERTED"
        else:
            action = "UNCHANGED"

        cv2.imwrite(str(output_path), image)

        total_images += 1

        # print(
        #     f"{rel_path} | "
        #     f"corners={[round(v,1) for v in corner_values]} | "
        #     f"white={white_count} "
        #     f"black={black_count} | "
        #     f"{action}"
        # )

    print("\n" + "=" * 60)
    print(f"Total images     : {total_images}")
    print(f"Inverted images  : {inverted_images}")
    print(f"Output directory : {OUTPUT_DIR}")
    print("=" * 60)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    process_dataset()
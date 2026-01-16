import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"


def apply_preprocessing(
    image_rgb,
    clip_limit = 0.5,
    tile_grid_size = (8, 8),
    mode = "lab"   # "lab", "green", "casp"
):
    clahe = cv2.createCLAHE(
        clipLimit = clip_limit,
        tileGridSize = tile_grid_size
    )

    if mode == "lab":
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    elif mode == "green":
        out = image_rgb.copy()
        out[:, :, 1] = clahe.apply(out[:, :, 1])
        return out

    elif mode == "casp":
        out = image_rgb.copy()

        # --- Green: CLAHE ---
        out[:, :, 1] = clahe.apply(out[:, :, 1])

        # --- Red & Blue: light blur + normalization ---
        for c in [0, 2]:  # Red, Blue
            channel = out[:, :, c]

            channel = cv2.GaussianBlur(
                channel,
                ksize = (3, 3),
                sigmaX = 0
            )

            min_val = channel.min()
            max_val = channel.max()

            if max_val > min_val:
                channel = (
                    (channel - min_val)
                    / (max_val - min_val)
                    * 255
                ).astype(channel.dtype)

            out[:, :, c] = channel

        return out

    else:
        raise ValueError("mode must be 'lab', 'green', or 'casp'")


# ---- IMAGE PATH ----
image_path = "IDRID/Original_Images/train/IDRiD_10.jpg"

image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# ---- Apply preprocessing variants ----
lab_clahe = apply_preprocessing(
    image_rgb,
    clip_limit = 2.0,
    tile_grid_size = (8, 8),
    mode = "lab"
)

green_clahe = apply_preprocessing(
    image_rgb,
    clip_limit = 2.0,
    tile_grid_size = (8, 8),
    mode = "green"
)

casp = apply_preprocessing(
    image_rgb,
    clip_limit = 2.0,
    tile_grid_size = (8, 8),
    mode = "casp"
)

# ---- Plot ----
plt.figure(figsize = (20, 5))

plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("CLAHE (L channel)")
plt.imshow(lab_clahe)
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("CLAHE (Green only)")
plt.imshow(green_clahe)
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("CASP (G-CLAHE, R/B stabilized)")
plt.imshow(casp)
plt.axis("off")

plt.tight_layout()
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

def apply_clahe_rgb(image_rgb, clip_limit = 0.5, tile_grid_size = (8, 8)):
    # RGB -> LAB
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit = clip_limit,
        tileGridSize = tile_grid_size
    )
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return rgb_clahe


def apply_clahe_green_only(image_rgb, clip_limit = 0.5, tile_grid_size = (8, 8)):
    # Extract green channel
    g = image_rgb[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit = clip_limit,
        tileGridSize = tile_grid_size
    )
    g_clahe = clahe.apply(g)

    # Replace green channel, keep R and B unchanged
    image_green_clahe = image_rgb.copy()
    image_green_clahe[:, :, 1] = g_clahe

    return image_green_clahe


# ---- IMAGE PATH ----
image_path = "IDRID/Original_Images/train/IDRiD_10.jpg"

image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Apply CLAHE variants
rgb_clahe = apply_clahe_rgb(
    image_rgb,
    clip_limit = 5.0,
    tile_grid_size = (8, 8)
)

green_clahe = apply_clahe_green_only(
    image_rgb,
    clip_limit = 5.0,
    tile_grid_size = (8, 8)
)

# ---- Plot ----
plt.figure(figsize = (15, 5))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("CLAHE (L channel)")
plt.imshow(rgb_clahe)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("CLAHE (Green channel only)")
plt.imshow(green_clahe)
plt.axis("off")

plt.tight_layout()
plt.show()

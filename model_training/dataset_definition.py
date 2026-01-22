"""
This file contains the dataset definition for fundus segmentation.
It is used directly by only `data_loader.py`.
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A

def center_crop_largest_square(image, **kwargs):
    h, w = image.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    return image[top : top + min_dim, left:left + min_dim]

def apply_clahe(
    image_rgb,
    clip_limit = 2.5,
    tile_grid_size = (8, 8),
    mode = "lab"   # "lab" or "green" or "casp"
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

        # --- Green channel: CLAHE ---
        out[:, :, 1] = clahe.apply(out[:, :, 1])

        # --- Red & Blue: light stabilization ---
        for c in [0, 2]:  # Red, Blue
            channel = out[:, :, c]

            # light blur (noise suppression)
            channel = cv2.GaussianBlur(channel, ksize = (3, 3), sigmaX = 0)
            # min–max normalization (avoid division by zero)
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:
                channel = ((channel - min_val) / (max_val - min_val) * 255).astype(channel.dtype)

            out[:, :, c] = channel

        return out
    else:
        raise ValueError("mode must be 'lab', 'green', or 'casp'")
    
class CLAHETransform(A.ImageOnlyTransform):
    def __init__(self, dataset):
        super().__init__(p = 1.0)
        self.dataset = dataset

    def apply(self, image, **params):
        if not self.dataset.use_clahe:
            return image

        return apply_clahe(
            image,
            clip_limit = self.dataset.clahe_clip,
            tile_grid_size = self.dataset.clahe_tile,
            mode = self.dataset.clahe_mode
        )

class FundusSegmentationDataset(Dataset):
    """
    Fundus Dataset
    - Center crop largest square
    - Resize to (dimensions, dimensions)

    image: (3, H, W)
    masks: (4, H, W)  # EX, HE, MA, SE
    """

    CLASSES = ["EX", "HE", "MA", "SE"]
    MASK_COLUMNS = {
        "EX": "ex_path",
        "HE": "he_path",
        "MA": "ma_path",
        "SE": "se_path",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        dimensions: int = 512,
        transform_type = "train",
        use_clahe: bool = False,
        clahe_clip: float = 2.5,
        clahe_tile: tuple = (8, 8),
        clahe_mode: str = "lab",
    ):
        self.df = df.reset_index(drop = True)
        self.dimensions = dimensions
        self.transform_type = transform_type

        self.use_clahe = use_clahe
        self.clahe_clip = float(clahe_clip)
        self.clahe_tile = clahe_tile
        self.clahe_mode = clahe_mode

        self.transforms = self._build_transforms()

    def _build_transforms(self):
        transforms = [
            A.Lambda(
                image = center_crop_largest_square,
                mask = center_crop_largest_square
            ),
            A.Resize(
                self.dimensions, 
                self.dimensions,
                interpolation = cv2.INTER_LINEAR,
                mask_interpolation = cv2.INTER_NEAREST
            ),
            CLAHETransform(self),
            A.Normalize(
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225),
            )
        ]

        if self.transform_type == "train":
            transforms.extend(
                [
                    A.HorizontalFlip(p = 0.5),
                    A.VerticalFlip(p = 0.5),
                    A.Affine(
                    translate_percent = 0.08,
                    scale = (0.88, 1.12),
                    rotate = (-15, 15),
                    border_mode = cv2.BORDER_CONSTANT,
                    fill = 0,
                    fill_mask = 0,
                    p = 0.7,
                )
                ]
            )

        return A.Compose(
            transforms,
            additional_targets = {
                "mask1": "mask",
                "mask2": "mask",
                "mask3": "mask",
                "mask4": "mask",
            },
            is_check_shapes = False,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]

        # ================= Image =================
        image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ================= Masks =================
        if row.dataset == "TJDR":
            # ---- TJDR: color-coded single annotation ----
            ann = cv2.imread(row.tjdr_ann_path, cv2.IMREAD_COLOR)
            ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)

            r, g, b = ann[..., 0], ann[..., 1], ann[..., 2]

            masks = [
                ((r == 128)   & (g == 0)   & (b == 0)).astype(np.uint8),  # EX (red)
                ((r == 0)   & (g == 128) & (b == 128)).astype(np.uint8),  # HE (green)
                ((r == 128) & (g == 128) & (b == 0)).astype(np.uint8),    # MA (yellow)
                ((r == 0) & (g == 0)   & (b == 128)).astype(np.uint8),    # SE (blue)
            ]
        else:
            # ---- DDR / IDRiD: per-class binary masks ----
            masks = []
            for cls in self.CLASSES:
                path = getattr(row, self.MASK_COLUMNS[cls])

                if pd.isna(path):
                    h, w = image.shape[:2]
                    mask = np.zeros((h, w), dtype = np.uint8)
                else:
                    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    mask = (mask > 0).astype(np.uint8)

                masks.append(mask)
        # ================= Transforms =================
        data = self.transforms(
            image = image,
            mask1 = masks[0],  # EX
            mask2 = masks[1],  # HE
            mask3 = masks[2],  # MA
            mask4 = masks[3],  # SE
        )
        image = torch.from_numpy(data["image"]).permute(2, 0, 1).float()
        masks = torch.stack(
            [(torch.from_numpy(data[f"mask{i+1}"]) > 0.5).float() for i in range(4)]
        )
        return image, masks

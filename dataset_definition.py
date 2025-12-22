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

class FundusSegmentationDataset(Dataset):
    """
    Helps produces the Fundus Dataset. There are 7 transform_type's. 
    We apply this class 7 times for each transform.

    Transforms:
    All transforms are applied to the masks and the image
    All transforms starting by cropping the largest square possible from the center of the image
    All transforms end by resizing to (self.dimensions, self.dimensions)
        t1 and test: Only Resize
        t2: Horizontal Flip
        t3: + or - 20% max zoom
        t4: Brightness and Contrast, Random Gamma
        t5: + or - 20 degree max rotation
        t6: Color Adjustments
        t7: Elastic Transform

    Default dimensions are 1024x1024. 
    Each mask is its own channel so the returned shapes are:
    image: (3, 1024, 1024)
    masks: (5, 1024, 1024)
    """
    def __init__(self, df: pd.DataFrame, dimensions: int = 1024, transform_type = None):
        self.df = df.reset_index(drop = True)
        self.dimensions = dimensions
        self.transform_type = transform_type
        self.transforms = self._build_transforms(transform_type)

    def _build_transforms(self, ttype):
        """ttype is any of (t1, t2, t3, t4, t5, t6, t7, test)"""
        if ttype is None:
            return None
        if ttype == 't1' or ttype == 'test':
            return A.Compose(
                [
                    A.Lambda(
                        image = center_crop_largest_square,
                        mask = center_crop_largest_square
                    ),
                    A.Resize(self.dimensions, self.dimensions)
            ],
                additional_targets = {
                        "mask1": "mask",
                        "mask2": "mask",
                        "mask3": "mask",
                        "mask4": "mask",
                        "mask5": "mask",
                },
                is_check_shapes = False
                )
        if ttype == "t2":
            return A.Compose(
                [
                    A.Lambda(
                        image = center_crop_largest_square,
                        mask = center_crop_largest_square
                    ),
                    A.HorizontalFlip(p = 1.0),
                    A.Resize(self.dimensions, self.dimensions)
                ], 
                additional_targets = {
                    "mask1": "mask",
                    "mask2": "mask",
                    "mask3": "mask",
                    "mask4": "mask",
                    "mask5": "mask",
                },
                is_check_shapes = False
            )
        if ttype == "t3":
            return A.Compose(
                [
                    A.Lambda(
                        image = center_crop_largest_square,
                        mask = center_crop_largest_square
                    ),
                    A.Affine(
                        scale = (0.80, 1.20),
                        translate_px = 0,
                        rotate = 0,
                        border_mode = cv2.BORDER_CONSTANT,
                        fill = 0,
                        fill_mask = 0,
                        p = 1.0
                    ),
                    A.Resize(self.dimensions, self.dimensions)
                ],
                additional_targets = {
                    "mask1": "mask",
                    "mask2": "mask",
                    "mask3": "mask",
                    "mask4": "mask",
                    "mask5": "mask",
                },
                is_check_shapes = False
            )
        if ttype == "t4":
            return A.Compose(
                [
                    A.Lambda(
                        image = center_crop_largest_square,
                        mask = center_crop_largest_square
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit = 0.2,
                        contrast_limit = 0.2,
                        p = 1.0
                    ),
                    A.RandomGamma(
                        gamma_limit = (80, 120),
                        p = 0.5
                    ),
                    A.Resize(self.dimensions, self.dimensions)
                ],
                additional_targets = {
                    "mask1": "mask",
                    "mask2": "mask",
                    "mask3": "mask",
                    "mask4": "mask",
                    "mask5": "mask",
                },
                is_check_shapes = False
            )
        if ttype == "t5":
            return A.Compose(
                [
                    A.Lambda(
                        image = center_crop_largest_square,
                        mask = center_crop_largest_square
                    ),
                    A.Rotate(
                        limit = 20,
                        border_mode = cv2.BORDER_CONSTANT,
                        fill = 0,
                        fill_mask = 0,
                        p = 1.0
                    ),
                    A.Resize(self.dimensions, self.dimensions)
                ],
                additional_targets = {
                    "mask1": "mask",
                    "mask2": "mask",
                    "mask3": "mask",
                    "mask4": "mask",
                    "mask5": "mask",
                },
                is_check_shapes = False
            )
        if ttype == "t6":
            return A.Compose(
                [
                    A.Lambda(
                        image = center_crop_largest_square,
                        mask = center_crop_largest_square
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit = 10,
                        sat_shift_limit = 20,
                        val_shift_limit = 10,
                        p = 0.8
                    ),
                    A.Resize(self.dimensions, self.dimensions)
                ],
                additional_targets = {
                    "mask1": "mask",
                    "mask2": "mask",
                    "mask3": "mask",
                    "mask4": "mask",
                    "mask5": "mask",
                },
                is_check_shapes = False
            )
        if ttype == "t7":
            return A.Compose(
                [
                    A.Lambda(
                        image = center_crop_largest_square,
                        mask = center_crop_largest_square
                    ),
                    A.ElasticTransform(
                        alpha = 100,
                        sigma = 10,
                        border_mode = cv2.BORDER_CONSTANT,
                        p = 1.0
                    ),
                    A.Resize(self.dimensions, self.dimensions)
                ],
                additional_targets = {
                    "mask1": "mask",
                    "mask2": "mask",
                    "mask3": "mask",
                    "mask4": "mask",
                    "mask5": "mask",
                },
                is_check_shapes = False
            )

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        row = self.df.loc[index]
        image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_paths = [row.ex_path, row.he_path, row.ma_path, row.se_path, row.od_path]

        masks = []
        for path in mask_paths:
            if path is None:
                height, width = image.shape[:2]
                mask = np.zeros((height, width), dtype = np.uint8)
            else:
                mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                mask = (mask > 0).astype(np.uint8)
            masks.append(mask)

        if self.transforms:
            data = self.transforms(
                image = image,
                mask1 = masks[0],
                mask2 = masks[1],
                mask3 = masks[2],
                mask4 = masks[3],
                mask5 = masks[4]
            )

            image = data["image"]
            masks = [
                data["mask1"],
                data["mask2"],
                data["mask3"],
                data["mask4"],
                data["mask5"]
            ]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.
        masks = torch.stack([torch.from_numpy(m) for m in masks]).float()

        return image, masks
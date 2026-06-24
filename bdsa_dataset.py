import os
import numpy as np  # Add this import
import albumentations
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import jpeg4py
from torchvision.transforms import transforms
from image_mask_dataset import GeneralDataModule, ImageMaskDataset, FtMaskDataset

class BCSSImageMaskDataset(ImageMaskDataset):
    def __init__(
            self,
            dataset_root: str,
            dataset_csv_path: str,
            data_type: str,
            val_fold_id: int,
            augmentation=None,
            data_ext: str =".jpg",
            dataset_mean=(0.485, 0.456, 0.406),
            dataset_std=(0.229, 0.224, 0.225),
            ignored_classes=None,
    ):
        super().__init__(
            dataset_root,
            dataset_csv_path,
            data_type,
            val_fold_id,
            augmentation,
            data_ext,
            dataset_mean,
            dataset_std,
            ignored_classes,
        )
        # Define mapping from grayscale values to class indices
        self.value_to_class = {
            191: 0,    # Exclude
            31: 1,   # class 1
            63: 2,   # class 2
            95: 3,   # class 3
            127: 4,  # class 4
            0  : 5   # class 5
        }



    def process_ignored_classes(self, mask):
        # First map the grayscale values to class indices
        mapped_mask = np.zeros_like(mask)
        for gray_value, class_idx in self.value_to_class.items():
            mapped_mask[mask == gray_value] = class_idx
            
        # Then handle ignored classes as in parent class
        if self.ignored_classes is not None:
            if not isinstance(self.ignored_classes, (list, tuple)):
                self.ignored_classes = [self.ignored_classes]
            for cls in self.ignored_classes:
                if cls != 0:
                    mapped_mask[mapped_mask == cls] = 0
        else:
            mapped_mask += 1
            
        return mapped_mask
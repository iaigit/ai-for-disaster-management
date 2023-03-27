from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import *

train_image_augmentations = A.Compose([
    A.Resize(image_size, image_size),
    A.Blur(),
    A.CoarseDropout(),
    A.Downscale(scale_min=0.5, scale_max=0.9),
    A.GaussNoise(),
    A.GridDistortion(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.ISONoise(),
    A.ImageCompression(),
    A.MotionBlur(),
    A.MultiplicativeNoise(),
    A.OpticalDistortion(),
    A.RandomBrightnessContrast(),
    A.RandomFog(),
    A.RandomGamma(),
    A.RandomRain(blur_value=3),
    A.Rotate(limit=(-30, 30)),
    A.Normalize(mean=mean_normalize, std=std_normalize),
    ToTensorV2()
])
test_image_augmentations = A.Compose([
    A.Resize(image_size[1], image_size[0]),
    A.Normalize(mean=mean_normalize, std=std_normalize),
    ToTensorV2()
])


class SemanticSegmentationDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        super().__init__()
        self.annotations_file = annotations_file
        self.transform = transform
        self.df = pd.read_csv(self.annotations_file, sep=" ", names=["image", "mask"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        image = np.array(Image.open(self.df.iloc[i][0]).convert("RGB"))
        mask  = np.array(Image.open(self.df.iloc[i][1]))
        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return (image, mask)
    

class SemanticSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, train_annotations_file, val_annotations_file,
                 test_annotations_file, batch_size, num_workers,
                 pin_memory):
        super().__init__()
        self.train_annotations_file = train_annotations_file
        self.val_annotations_file = val_annotations_file
        self.test_annotations_file = test_annotations_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def main_dataloader(self, mode):
        shuffle = False
        augmentations = test_image_augmentations
        if mode == "train":
            shuffle = True
            augmentations = train_image_augmentations
            annotations_file = self.train_annotations_file
        elif mode == "val":
            annotations_file = self.val_annotations_file
        else:
            annotations_file = self.test_annotations_file
        dataset = SemanticSegmentationDataset(
            annotations_file,
            augmentations
        )
        dataloader = DataLoader(dataset=dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            shuffle=shuffle,
        )
        return dataloader

    def train_dataloader(self):
        return self.main_dataloader("train")

    def val_dataloader(self):
        return self.main_dataloader("val")

    def test_dataloader(self):
        return self.main_dataloader("test")
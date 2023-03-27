from pytorch_lightning.callbacks import ModelCheckpoint
from models.base_model import SemanticSegmentationLightningModule
from data.dataset import SemanticSegmentationDataModule
from config import *

import pytorch_lightning as pl

def main():
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir, 
            monitor="val_iou_score", 
            mode="max", 
            save_last=True
        )
    ]

    data_module = SemanticSegmentationDataModule(
        train_annotations_file,
        val_annotations_file,
        test_annotations_file,
        batch_size,
        num_workers,
        pin_memory
    )

    lightning_module = SemanticSegmentationLightningModule(
        learning_rate,
        weight_decay,
        mode_segmentation,
        num_classes,
        mask_threshold
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        precision=16 if fp16 else 32,
        gradient_clip_val=gradient_clip,
        default_root_dir=trainer_root_dir,
        devices=-1,
        accelerator="auto",
        auto_select_gpus=True,
    )
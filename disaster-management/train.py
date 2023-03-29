from pytorch_lightning.callbacks import ModelCheckpoint
from models.base_model import SemanticSegmentationLightningModule
from data.dataset import SemanticSegmentationDataModule
from config import *

import pytorch_lightning as pl

import wandb

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
        gpus=0,
        accelerator="auto",
    )

    trainer.fit(
    model=lightning_module, 
    train_dataloaders=data_module,
        )
    
    trainer.test(dataloaders=data_module, ckpt_path=trainer.checkpoint_callback.best_model_path)

    lightning_module.stop_wandb()

    lightning_module = SemanticSegmentationLightningModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        mode_segmentation=mode_segmentation,
        num_classes=num_classes,
        mask_threshold=mask_threshold
    )

    #push the model to the wandb server
    
if __name__ == '__main__':
    main()
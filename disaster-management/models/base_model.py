import time
import numpy as np
import torch
import torchmetrics as tm
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import metrics as seg_metrics
from config import *
import wandb


class SemanticSegmentationLightningModule(pl.LightningModule):
    def __init__(self, learning_rate, weight_decay, mode_segmentation, num_classes,
                 mask_threshold, model_type=model_type, criterion_type=criterion_type):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.mode_segmentation = mode_segmentation
        self.num_classes = num_classes
        self.mask_threshold = mask_threshold
        
        if model_type == "unet":
            from . import unet
            self.model = unet.UNET()
        elif model_type == "att_squeeze_unet":
            from . import att_squeeze_unet
            self.model = att_squeeze_unet.AttSqueezeUnet()
        elif model_type == "att_unet":
            from . import att_unet
            self.model = att_unet.AttUNet()
        elif model_type == "dc_unet":
            from . import dc_unet
            self.model = dc_unet.DcUnet()
        elif model_type == "res_unet":
            from . import res_unet
            self.model = res_unet.ResUnet()
        elif model_type == "res_unet_plus_plus":
            from . import res_unet_plus_plus
            self.model = res_unet_plus_plus.ResUnetPlusPlus()
        elif model_type == "sa_unet":
            from . import sa_unet
            self.model = sa_unet.SA_UNet()
        else:
            raise ValueError("Model type not found")
        
        if criterion_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif criterion_type == "dice":
            from .loss import dice_loss
            self.criterion = dice_loss.BinaryDiceLoss()
        
        wandb.init(project="disaster-management", config={
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "mode_segmentation": mode_segmentation,
            "num_classes": num_classes,
            "mask_threshold": mask_threshold,
            "model_type": model_type,
            "criterion_type": criterion_type,
            "epochs": epochs,
            "image_size": image_size,
            "mean normalize": mean_normalize,
            "std normalize": std_normalize,
            "data": data_name
        })

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda x: (((1 + np.cos(x * np.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1
        )
        return {
           'optimizer': optimizer,
           'lr_scheduler': lr_scheduler,
           'monitor': 'val_iou_score'
       }

    def forward(self, x):
        return self.model(x)

    def calculate_metrics(self, predict_mask, mask):
        tp, fp, fn, tn = seg_metrics.get_stats(
            predict_mask, mask,
            mode=self.mode_segmentation,
            threshold=self.mask_threshold,
            num_classes=self.num_classes
        )
        iou_score = seg_metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = tm.functional.classification.binary_f1_score(predict_mask, mask, threshold=self.mask_threshold)
        accuracy = tm.functional.classification.binary_accuracy(predict_mask, mask, threshold=self.mask_threshold)
        precision = tm.functional.classification.binary_precision(predict_mask, mask, threshold=self.mask_threshold)
        recall = tm.functional.classification.binary_recall(predict_mask, mask, threshold=self.mask_threshold)
        dice_score = tm.functional.classification.dice(predict_mask, mask, threshold=self.mask_threshold, multiclass=False, average="micro")
        return iou_score, f1_score, accuracy, precision, recall, dice_score

    def main_loop_step(self, batch, batch_idx, mode):
        image, mask = batch
        mask = mask.unsqueeze(1)
        start_time = time.perf_counter()
        preds_mask = self(image)
        inference_time = time.perf_counter() - start_time
        loss = self.criterion(preds_mask, mask.float())
        iou_score, f1_score, accuracy, precision, recall, dice_score = self.calculate_metrics(preds_mask, mask)

        on_step = False if mode == "train" else True
        self.log(f'{mode}_iou_score', iou_score, on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_f1_score', f1_score, on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_accuracy', accuracy, on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_precision', precision, on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_recall', recall, on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_dice_score', dice_score, on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f'{mode}_inference_time', inference_time, on_step=on_step, on_epoch=True, prog_bar=True)

        wandb.log({
            f"{mode}_iou_score": iou_score,
            f"{mode}_f1_score": f1_score,
            f"{mode}_accuracy": accuracy,
            f"{mode}_precision": precision,
            f"{mode}_recall": recall,
            f"{mode}_dice_score": dice_score,
            f"{mode}_inference_time": inference_time
        })

        return loss

    def training_step(self, batch, batch_idx):
        return self.main_loop_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.main_loop_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.main_loop_step(batch, batch_idx, "test")
    
    def stop_wandb(self):
        wandb.finish()
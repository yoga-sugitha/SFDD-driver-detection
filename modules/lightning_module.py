"""
Lightning training module
"""
import torch
import torch.nn as nn
import lightning as L
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, 
    MulticlassPrecision, MulticlassRecall
)
from typing import List, Dict, Any
from models.factory import create_model

class LightningModule(L.LightningModule):
    """
    Lightning module for training ResNet models
    
    Args:
        model_name: Name of model architecture
        model_hparams: Model hyperparameters
        optimizer_name: Name of optimizer
        optimizer_hparams: Optimizer hyperparameters
        class_names: List of class names for logging
    """
    def __init__(
        self,
        model_name: str,
        model_hparams: Dict[str, Any],
        optimizer_name: str,
        optimizer_hparams: Dict[str, Any],
        class_names: List[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_names"])
        
        # Create model
        self.model = create_model(model_name, model_hparams)
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_classes = model_hparams.get('num_classes', 10)
        self.class_names = class_names or [f"Class {i}" for i in range(self.num_classes)]
        
        # Setup metrics
        metrics = MetricCollection({
            "acc": MulticlassAccuracy(num_classes=self.num_classes),
            "precision": MulticlassPrecision(num_classes=self.num_classes, average="macro"),
            "recall": MulticlassRecall(num_classes=self.num_classes, average="macro"),
            "f1_macro": MulticlassF1Score(num_classes=self.num_classes, average="macro"),
            "f1_weighted": MulticlassF1Score(num_classes=self.num_classes, average="weighted"),
        })
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        
        # Storage for test predictions
        self.test_preds = []
        self.test_targets = []
    
    def forward(self, imgs):
        return self.model(imgs)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer_name = self.hparams.optimizer_name.lower()
        
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optimizer_hparams)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            print(f"Warning: Unknown optimizer {optimizer_name}, using Adam")
            optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optimizer_hparams)
        
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=10
            ),
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]
    
    def _shared_step(self, batch):
        """Shared step for train/val/test"""
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        return loss, preds, y
    
    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.val_metrics.update(preds, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
    
    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()
    
    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.test_metrics.update(preds, y)
        self.test_preds.append(preds.argmax(dim=1).detach())
        self.test_targets.append(y.detach())
    
    def on_test_epoch_end(self):
        """Compute final test metrics"""
        results = self.test_metrics.compute()
        self.log_dict(results, sync_dist=True)
        
        # Store predictions for later visualization
        if len(self.test_preds) > 0:
            self.all_test_preds = torch.cat(self.test_preds, dim=0)
            self.all_test_targets = torch.cat(self.test_targets, dim=0)
        
        self.test_metrics.reset()

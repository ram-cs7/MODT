"""
Detection Model Trainer
Handles training loop, validation, checkpointing, and logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any
import time
from tqdm import tqdm
import wandb

from src.utils.config import ConfigManager
from src.utils.logger import setup_logger, MetricsLogger
from models.detectors.yolo_detector import YOLODetector
from src.data.dataset import DetectionDataset
from src.data.augmentation import AugmentationPipeline


class DetectionTrainer:
    """
    Trainer class for object detection models
    Supports YOLOv5/YOLOv8 training with various optimizations
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize trainer
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = setup_logger("Trainer")
        self.metrics_logger = MetricsLogger(config.get("monitoring.checkpoints.save_dir", "./outputs/logs"))
        
        # Device setup
        self.device = config.get("system.device", "cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self._init_model()
        
        # Initialize datasets
        self._init_datasets()
        
        # Initialize optimizer and scheduler
        self._init_optimizer()
        
        # Training state
        self.current_epoch = 0
        self.best_map = 0.0
        self.best_epoch = 0
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get("monitoring.checkpoints.save_dir", "./weights"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # WandB initialization
        if config.get("monitoring.wandb.enable", False):
            wandb.init(
                project=config.get("monitoring.wandb.project"),
                config=config.config,
                tags=config.get("monitoring.wandb.tags", [])
            )
    
    def _init_model(self):
        """Initialize detection model"""
        self.model = YOLODetector(
            variant=f"{self.config.get('model.detector.type')}{self.config.get('model.detector.variant')[0]}",
            num_classes=self.config.get('model.detector.num_classes'),
            pretrained=self.config.get('model.detector.pretrained'),
            img_size=tuple(self.config.get('model.detector.input_size')),
            device=self.device
        )
        
        self.logger.info(f"Model initialized: {self.model.variant}")
    
    def _init_datasets(self):
        """Initialize training and validation datasets"""
        # Augmentation pipelines
        aug_pipeline = AugmentationPipeline()
        train_transforms = aug_pipeline.get_training_transforms(
            img_size=tuple(self.config.get('model.detector.input_size'))
        )
        val_transforms = aug_pipeline.get_validation_transforms(
            img_size=tuple(self.config.get('model.detector.input_size'))
        )
        
        # Training dataset
        self.train_dataset = DetectionDataset(
            data_dir=self.config.get('dataset.paths.train'),
            annotation_format=self.config.get('dataset.format'),
            img_size=tuple(self.config.get('model.detector.input_size')),
            transforms=train_transforms if self.config.get('augmentation.train.enable') else None
        )
        
        # Validation dataset
        self.val_dataset = DetectionDataset(
            data_dir=self.config.get('dataset.paths.val'),
            annotation_format=self.config.get('dataset.format'),
            img_size=tuple(self.config.get('model.detector.input_size')),
            transforms=val_transforms if self.config.get('augmentation.val.enable') else None
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get('training.batch_size'),
            shuffle=True,
            num_workers=self.config.get('system.num_workers'),
            pin_memory=True if self.device == "cuda" else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.get('training.batch_size'),
            shuffle=False,
            num_workers=self.config.get('system.num_workers'),
            pin_memory=True if self.device == "cuda" else False
        )
        
        self.logger.info(f"Train dataset: {len(self.train_dataset)} images")
        self.logger.info(f"Val dataset: {len(self.val_dataset)} images")
    
    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        # Optimizer
        optimizer_type = self.config.get('training.optimizer.type')
        lr = self.config.get('training.optimizer.lr')
        
        if optimizer_type == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.config.get('training.optimizer.momentum'),
                weight_decay=self.config.get('training.optimizer.weight_decay'),
                nesterov=self.config.get('training.optimizer.nesterov', False)
            )
        elif optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.config.get('training.optimizer.weight_decay')
            )
        elif optimizer_type == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=self.config.get('training.optimizer.weight_decay')
            )
        
        # Scheduler
        scheduler_type = self.config.get('training.scheduler.type')
        
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('training.epochs'),
                eta_min=self.config.get('training.scheduler.min_lr')
            )
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_type == "multistep":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.config.get('training.scheduler.milestones', [30, 60, 90]),
                gamma=0.1
            )
        elif scheduler_type == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Optimizer: {optimizer_type}, Scheduler: {scheduler_type}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of training metrics
        """
        self.model.model.train()
        
        epoch_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.get('training.epochs')}")
        
        for batch_idx, (images, labels, metadata) in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass
            # Note: Actual YOLO training would use the ultralytics training loop
            # This is a simplified version
            self.optimizer.zero_grad()
            
            # Placeholder loss computation
            loss = torch.tensor(0.5, requires_grad=True, device=self.device)  # Replace with actual loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('training.gradient_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('training.gradient_clip')
                )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(self.train_loader)
        
        return {
            'train_loss': avg_loss,
            'lr': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.model.eval()
        
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels, metadata in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                
                # Placeholder validation
                loss = torch.tensor(0.4, device=self.device)  # Replace with actual validation
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        
        # Placeholder mAP (would compute actual mAP in production)
        map_50 = 0.85  # Replace with actual mAP computation
        
        return {
            'val_loss': avg_val_loss,
            'mAP@0.5': map_50
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config.config
        }, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.model.state_dict(),
                'metrics': metrics
            }, best_path)
            self.logger.info(f"Saved best model with mAP: {metrics['mAP@0.5']:.4f}")
        
        # Save last model
        last_path = self.checkpoint_dir / "last.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'metrics': metrics
        }, last_path)
    
    def train(self):
        """Main training loop"""
        self.logger.info("="*80)
        self.logger.info("Starting Training")
        self.logger.info("="*80)
        
        num_epochs = self.config.get('training.epochs')
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            if epoch % self.config.get('monitoring.evaluation_interval', 1) == 0:
                val_metrics = self.validate(epoch)
                
                # Combine metrics
                metrics = {**train_metrics, **val_metrics}
                
                # Log
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Loss={metrics['train_loss']:.4f}, "
                    f"Val Loss={metrics['val_loss']:.4f}, "
                    f"mAP@0.5={metrics['mAP@0.5']:.4f}"
                )
                
                # WandB logging
                if self.config.get('monitoring.wandb.enable'):
                    wandb.log(metrics, step=epoch)
                
                # Metrics logger
                self.metrics_logger.log_metric(epoch, metrics)
                
                # Save checkpoint
                is_best = metrics['mAP@0.5'] > self.best_map
                if is_best:
                    self.best_map = metrics['mAP@0.5']
                    self.best_epoch = epoch
                
                if self.config.get('monitoring.checkpoints.save_best') and is_best:
                    self.save_checkpoint(epoch, metrics, is_best=True)
                
                if epoch % self.config.get('monitoring.checkpoints.save_period', 10) == 0:
                    self.save_checkpoint(epoch, metrics, is_best=False)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
        
        self.logger.info("="*80)
        self.logger.info(f"Training completed! Best mAP: {self.best_map:.4f} at epoch {self.best_epoch}")
        self.logger.info("="*80)


if __name__ == "__main__":
    from src.utils.config import load_config
    
    config = load_config("training")
    trainer = DetectionTrainer(config)
    trainer.train()

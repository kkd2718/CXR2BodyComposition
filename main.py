from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer, SGD, Adam, AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.utils import PathManager, RandomSeeder, CheckpointManager
from utils.dataset import BodyCompositionDevelop
from utils.ldl import LabelDistributionLearning, MeanVarianceLoss
from utils.config_base import parse_arguments, BaseConfig

@dataclass
class ModelMetrics:
    loss: float
    mae: float
    epoch: int


class ModelTrainer:
    def __init__(self, config: BaseConfig, model: nn.Module, 
                 optimizer: Optimizer, device: torch.device,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.ce_loss = nn.CrossEntropyLoss()
        self.setup_loss_functions()

    def train_epoch(self, epoch: int, train_loader: DataLoader, writer: SummaryWriter) -> None:
        self.model.train()
        metrics = defaultdict(list)
        
        for batch_idx, data in enumerate(train_loader, 1):
            loss, batch_metrics = self.process_batch(data, training=True)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            for k, v in batch_metrics.items():
                metrics[k].append(v)
                
            self._log_progress('train', epoch, batch_idx, len(train_loader), metrics)
            
        self._write_metrics(writer, metrics, epoch)

    @torch.no_grad()
    def validate(self, epoch: int, valid_loader: DataLoader, writer: SummaryWriter) -> ModelMetrics:
        self.model.eval()
        metrics = defaultdict(list)
        
        for batch_idx, data in enumerate(valid_loader, 1):
            loss, batch_metrics = self.process_batch(data, training=False)
            
            for k, v in batch_metrics.items():
                metrics[k].append(v)
                
            self._log_progress('valid', epoch, batch_idx, len(valid_loader), metrics)
            
        self._write_metrics(writer, metrics, epoch)
        
        mae_key = f'mae_{self.config.task}'
        
        if self.scheduler and self.config.scheduler == 'on':
            self.scheduler.step(np.mean(metrics['loss']))
        
        return ModelMetrics(
            loss=np.mean(metrics['loss']),
            mae=np.mean(metrics[mae_key]),
            epoch=epoch
        )
        
    def setup_loss_functions(self):
        self.meanvar_losses = {
            'task': MeanVarianceLoss(task=self.config.task),
            'age': MeanVarianceLoss(task='age'),
            'height': MeanVarianceLoss(task='height'),
            'weight': MeanVarianceLoss(task='weight')
        }

    def process_batch(self, data: Dict[str, Tensor], training: bool = True) -> Tuple[Tensor, Dict[str, float]]:
        inputs = {k: v.to(self.device, dtype=torch.float if k == 'input' else torch.long)
                 for k, v in data.items() if isinstance(v, Tensor)}
        
        outputs = self.model(inputs['input'])
        if self.config.backbone == 'inception':
            outputs = outputs[0]
            
        if self.config.adjust == 'base':
            return self._process_base(outputs, inputs)
        elif self.config.adjust == 'a1':
            return self._process_a1(outputs, inputs)
        elif self.config.adjust == 'a2':
            return self._process_a2(outputs, inputs)
        else:
            raise ValueError(f"Unknown adjustment type: {self.config.adjust}")
        
    def _process_base(self, outputs: Tensor, inputs: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        target = inputs[self.config.task]
        ce_loss = self.ce_loss(outputs, target)
        mean_loss, var_loss = self.meanvar_losses['task'](outputs, target)
        
        loss = (self.config.lambda_ce * ce_loss + 
                self.config.lambda_mean * mean_loss +
                self.config.lambda_var * var_loss)
                
        with torch.no_grad():
            mae = LabelDistributionLearning.calculate_mae(outputs.detach().cpu(),
                                                    target.cpu(),
                                                    self.config.task)
                                                    
        metrics = {
            'loss': loss.item(),
            f'mae_{self.config.task}': mae.item()
        }

        return loss, metrics
    
    def _process_a1(self, outputs: List[Tensor], inputs: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        target = inputs[self.config.task]
        age = inputs['age']
        sex = inputs['sex']
        
        ce_task = self.ce_loss(outputs[0], target)
        ce_age = self.ce_loss(outputs[1], age)
        ce_sex = self.ce_loss(outputs[2], sex)
        
        mean_task, var_task = self.meanvar_losses['task'](outputs[0], target)
        mean_age, var_age = self.meanvar_losses['age'](outputs[1], age)
        
        loss = (
            self.config.lambda_ce * ce_task +
            self.config.lambda_mean * mean_task +
            self.config.lambda_var * var_task +
            self.config.lambda_ce * (ce_age + ce_sex) +
            self.config.lambda_mean * mean_age +
            self.config.lambda_var * var_age
        )
        
        with torch.no_grad():
            mae_target = LabelDistributionLearning.calculate_mae(
                outputs[0].detach().cpu(),
                target.cpu(),
                self.config.task
            )
            mae_age = LabelDistributionLearning.calculate_mae(
                outputs[1].detach().cpu(),
                age.cpu(),
                'age'
            )
            _, preds = torch.max(outputs[2], 1)
            sex_accuracy = torch.sum(preds == sex).item() / sex.size(0)
        
        metrics = {
            'loss': loss.item(),
            f'mae_{self.config.task}': mae_target.item(),
            'mae_age': mae_age.item(),
            'sex_accuracy': sex_accuracy
        }
        
        return loss, metrics
    
    def _process_a2(self, outputs: List[Tensor], inputs: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        target = inputs[self.config.task]
        age = inputs['age']
        sex = inputs['sex']
        height = inputs['height']
        weight = inputs['weight']
        
        ce_task = self.ce_loss(outputs[0], target)
        ce_age = self.ce_loss(outputs[1], age)
        ce_sex = self.ce_loss(outputs[2], sex)
        ce_height = self.ce_loss(outputs[3], height)
        ce_weight = self.ce_loss(outputs[4], weight)
        
        mean_task, var_task = self.meanvar_losses['task'](outputs[0], target)
        mean_age, var_age = self.meanvar_losses['age'](outputs[1], age)
        mean_height, var_height = self.meanvar_losses['height'](outputs[3], height)
        mean_weight, var_weight = self.meanvar_losses['weight'](outputs[4], weight)
        
        loss = (
            self.config.lambda_ce * ce_task +
            self.config.lambda_mean * mean_task +
            self.config.lambda_var * var_task +
            self.config.lambda_ce * (ce_age + ce_sex) +
            self.config.lambda_mean * mean_age +
            self.config.lambda_var * var_age +
            self.config.lambda_ce * (ce_height + ce_weight) +
            self.config.lambda_mean * (mean_height + mean_weight) +
            self.config.lambda_var * (var_height + var_weight)
        )
        
        with torch.no_grad():
            mae_target = LabelDistributionLearning.calculate_mae(
                outputs[0].detach().cpu(),
                target.cpu(),
                self.config.task
            )
            mae_age = LabelDistributionLearning.calculate_mae(
                outputs[1].detach().cpu(),
                age.cpu(),
                'age'
            )
            _, preds = torch.max(outputs[2], 1)
            sex_accuracy = torch.sum(preds == sex).item() / sex.size(0)
            mae_height = LabelDistributionLearning.calculate_mae(
                outputs[3].detach().cpu(),
                height.cpu(),
                'height'
            )
            mae_weight = LabelDistributionLearning.calculate_mae(
                outputs[4].detach().cpu(),
                weight.cpu(),
                'weight'
            )
        
        metrics = {
            'loss': loss.item(),
            f'mae_{self.config.task}': mae_target.item(),
            'mae_age': mae_age.item(),
            'sex_accuracy': sex_accuracy,
            'mae_height': mae_height.item(),
            'mae_weight': mae_weight.item()
        }
        
        return loss, metrics
    
    def _write_metrics(self, writer: SummaryWriter, metrics: Dict[str, list], epoch: int) -> None:
        """Write metrics to tensorboard"""
        try:
            for key, values in metrics.items():
                writer.add_scalar(f"{key}", np.mean(values), epoch)
        except Exception as e:
            print(f"Warning: Failed to write metrics - {str(e)}")

    def _log_progress(
        self, 
        phase: str,
        epoch: int, 
        batch_idx: int, 
        total_batches: int, 
        metrics: Dict[str, List[float]]
    ) -> None:
        current_metrics = {k: np.mean(v) for k, v in metrics.items()}
        
        log_str = f'\r{phase}: Epoch {epoch:04d}/{self.config.num_epoch:04d}|'
        log_str += f'Batch {batch_idx:04d}/{total_batches:04d}|'
        
        metric_strs = [f'Loss {current_metrics["loss"]:.4f}']
        metric_strs.append(f'{self.config.task} {current_metrics[f"mae_{self.config.task}"]:.4f}')
        
        if self.config.adjust in ['a1', 'a2']:
            metric_strs.append(f'Age {current_metrics["mae_age"]:.4f}')
            metric_strs.append(f'Sex {current_metrics["sex_accuracy"]:.4f}')
            
        if self.config.adjust == 'a2':
            metric_strs.append(f'Height {current_metrics["mae_height"]:.4f}')
            metric_strs.append(f'Weight {current_metrics["mae_weight"]:.4f}')
            
        log_str += '|'.join(metric_strs)
        
        print(log_str, end='', flush=True)
        if batch_idx == total_batches:
            print()


class Training:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.device = self._setup_device()
        self._log_gpu_stats()
        self.best_metrics = ModelMetrics(
            loss=float('inf'),
            mae=float('inf'),
            epoch=0
        )

    def train(self) -> None:
        """Main training loop"""
        for epoch in range(self.config.start_epoch + 1, self.config.num_epoch + 1):
            self.trainer.train_epoch(epoch, self.train_loader, self.writer_train)
            metrics = self.trainer.validate(epoch, self.valid_loader, self.writer_valid)
            
            if metrics.loss < self.best_metrics.loss:
                self.best_metrics = metrics
                CheckpointManager.save(
                    self.ckpt_dir,
                    self.model,
                    self.optimizer,
                    self.best_metrics.epoch,
                    self.best_metrics.loss
                )
                print(f"Best valid loss {metrics.loss:.4f} at epoch {metrics.epoch}")
            
            if self.config.scheduler == 'on':
                self.scheduler.step(metrics.loss)
                
        print(f"Best validation loss {self.best_metrics.loss:.4f} at epoch {self.best_metrics.epoch}")
        print(f"MAE: {self.best_metrics.mae:.4f} at epoch {self.best_metrics.epoch}")

    def _log_gpu_stats(self) -> None:
        """Log GPU memory usage"""
        print('Current GPU:', torch.cuda.current_device())
        if self.device.type == 'cuda':
            print('Allocated:', round(torch.cuda.memory_allocated(self.config.gpu)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(self.config.gpu)/1024**3,1), 'GB')
        print('Device:', self.device)

    @staticmethod
    def calculate_parameters(model: nn.Module) -> float:
        """Calculate total parameters in millions"""
        return sum(param.numel() for param in model.parameters())/1000000.0

    @staticmethod
    def calculate_trainables(model: nn.Module) -> float:
        """Calculate trainable parameters in millions"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000.0
        
    def _setup_device(self) -> torch.device:
        device = torch.device(f'cuda:{self.config.gpu}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
        return device
        
    def setup(self) -> None:
        RandomSeeder.seed_all(self.config.seed)
        print("Random seed fixed:", self.config.seed)
        
        self._setup_data()
        self._setup_model()
        print("Model setting completed: model -", self.config.backbone, 
              ", adjustment -", self.config.adjust)
        print(f"Total Parameter: {self.calculate_parameters(self.model):.2f}M")
        print(f"Trainable Parameter: {self.calculate_trainables(self.model):.2f}M")
        self._setup_optimizer()
        self._setup_logging()
        self.load_checkpoint()
        
        self.trainer = ModelTrainer(
            config=self.config,
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            scheduler=self.scheduler if hasattr(self, 'scheduler') else None
        )

    def cleanup(self) -> None:
        try:
            if hasattr(self, 'writer_train'):
                self.writer_train.close()
            if hasattr(self, 'writer_valid'):
                self.writer_valid.close()
        except Exception as e:
            print(f"Warning: Error during cleanup - {str(e)}")

    def _setup_data(self) -> None:
        train_dataset = BodyCompositionDevelop(mode='train')
        valid_dataset = BodyCompositionDevelop(mode='tuning')
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch,
            shuffle=True,
            num_workers=self.config.cpu
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.batch,
            shuffle=True,
            num_workers=self.config.cpu
        )

    def _setup_model(self) -> None:
        if self.config.backbone == 'inception':
            import model.inception as backbone
            self.model = backbone.Inception3(adjust=self.config.adjust, task=self.config.task)
            self.model.to(self.device)

    def _setup_optimizer(self) -> None:
        optimizer_classes = {
            'sgd': lambda: SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9),
            'adam': lambda: Adam(self.model.parameters(), lr=self.config.lr),
            'adamw': lambda: AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=0.01)
        }
        
        self.optimizer = optimizer_classes[self.config.optim]()
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2, eta_min=0
        )

    def _setup_logging(self) -> None:
        self.log_dir = PathManager.get_log_dir(self.config)
        self.ckpt_dir = PathManager.get_checkpoint_dir(self.config)
        
        self.writer_train = SummaryWriter(log_dir=self.log_dir / 'train')
        self.writer_valid = SummaryWriter(log_dir=self.log_dir / 'valid')

    def _create_log_path(self, base_dir: Path) -> Path:
        log_path = base_dir / self.config.task / self.config.backbone / self.config.optim
        
        params = [
            f'adjustment_{self.config.adjust}',
            f'meanvar_{self.config.meanvar}',
            f'lr_{self.config.lr}',
            f'scheduler_{self.config.scheduler}',
            f'lambdaCE_{self.config.lambda_ce}',
            f'lambdaMean_{self.config.lambda_mean}',
            f'lambdaVar_{self.config.lambda_var}',
            'log'
        ]
        
        log_path = log_path.joinpath(*params)
        log_path.mkdir(parents=True, exist_ok=True)
        
        return log_path
    
    def _create_checkpoint_path(self, base_dir: Path) -> Path:
        ckpt_path = base_dir / self.config.task / self.config.backbone / self.config.optim
        
        params = [
            f'adjustment_{self.config.adjust}',
            f'meanvar_{self.config.meanvar}',
            f'lr_{self.config.lr}',
            f'scheduler_{self.config.scheduler}',
            f'lambdaCE_{self.config.lambda_ce}',
            f'lambdaMean_{self.config.lambda_mean}',
            f'lambdaVar_{self.config.lambda_var}',
            'ckpt'
        ]
        
        ckpt_path = ckpt_path.joinpath(*params)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        
        return ckpt_path
    
    def load_checkpoint(self) -> None:
        if self.config.resume != 'true':
            return
            
        try:
            ckpt_files = sorted(self.ckpt_dir.glob('*.pth'))
            if not ckpt_files:
                raise FileNotFoundError("No checkpoint files found")
                
            latest_ckpt = ckpt_files[-1]
            load_dict = torch.load(
                latest_ckpt,
                map_location=f"cuda:{self.config.gpu}"
            )
            
            self.config.start_epoch = load_dict['epoch']
            self.best_metrics = ModelMetrics(
                loss=load_dict['loss'],
                mae=float('inf'),
                epoch=load_dict['epoch']
            )
            
            self.model.load_state_dict(load_dict['model'])
            self.optimizer.load_state_dict(load_dict['optim'])
            
            print(f"Loaded checkpoint from {latest_ckpt}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            print("Starting training from scratch")
            self.config.start_epoch = 0
            self.best_metrics = ModelMetrics(
                loss=float('inf'),
                mae=float('inf'),
                epoch=0
            )
            

def main():
    config = parse_arguments()
    training = Training(config)
    
    try:
        training.setup()
        training.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    finally:
        training.cleanup()

if __name__ == '__main__':
    main()
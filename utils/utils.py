from pathlib import Path
from typing import List, Union, Dict, Tuple
import os
import random
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from dataclasses import dataclass
from .config_base import BaseConfig

@dataclass
class TaskRanges:
    """Class information for different tasks"""
    RANGES: Dict[str, Tuple[int, int, int]] = {
        'age': (0, 100, 101),
        'height': (120, 210, 901),
        'weight': (25, 170, 1451),
        'fat': (0, 90, 901),
        'mmwhole': (15, 100, 851),
        'mmappend': (5, 70, 651),
        'wofat': (15, 110, 951),
        'waist': (50, 170, 1201)
    }


class PathManager:
    @staticmethod
    def get_data_dir() -> Path:
        return Path(os.getenv('DATA_DIR', 'default/path'))

    @staticmethod
    def get_record_dir() -> Path:
        return Path('record_path')
    
    @staticmethod
    def get_checkpoint_dir(config: BaseConfig) -> Path:
        return (PathManager.get_record_dir() / 
                config.task / config.backbone / config.optim /
                f'adjustment_{config.adjust}' / f'meanvar_{config.meanvar}' /
                f'lr_{config.lr}' / f'scheduler_{config.scheduler}' /
                f'lambdaCE_{config.lambda_ce}' / f'lambdaMean_{config.lambda_mean}' /
                f'lambdaVar_{config.lambda_var}' / 'ckpt')
                
    @staticmethod
    def get_log_dir(config: BaseConfig) -> Path:
        return (PathManager.get_record_dir() / 
                config.task / config.backbone / config.optim /
                f'adjustment_{config.adjust}' / f'meanvar_{config.meanvar}' /
                f'lr_{config.lr}' / f'scheduler_{config.scheduler}' /
                f'lambdaCE_{config.lambda_ce}' / f'lambdaMean_{config.lambda_mean}' /
                f'lambdaVar_{config.lambda_var}' / 'log')
    

class CheckpointManager:
    @staticmethod
    def save(
        ckpt_dir: Union[str, Path],
        model: nn.Module,
        optim: Optimizer,
        best_epoch: int,
        best_loss: float
    ) -> None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract state dict handling DataParallel case
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        
        checkpoint = {
            'model': model_state,
            'optim': optim.state_dict(),
            'loss': best_loss,
            'epoch': best_epoch
        }
        
        save_path = ckpt_dir / f'BiologicalAge{best_epoch}.pth'
        torch.save(checkpoint, save_path)


class RandomSeeder:
    @staticmethod
    def seed_all(seed: int) -> None:
        # PyTorch random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # CuDNN settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Other libraries
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


class AverageMeter:
    """Computes and stores the average and current value"""    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
    
    def __str__(self) -> str:
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Displays training progress"""
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save(ckpt_dir: Union[str, Path], model: nn.Module, optim: Optimizer, 
         best_epoch: int, best_loss: float) -> None:
    CheckpointManager.save(ckpt_dir, model, optim, best_epoch, best_loss)

def rand_fix(random_seed: int) -> None:
    RandomSeeder.seed_all(random_seed)
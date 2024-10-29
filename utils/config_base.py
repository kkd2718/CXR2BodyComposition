from dataclasses import dataclass
from typing import Optional
import argparse

@dataclass
class BaseConfig:
    """Base configuration parameters"""
    gpu: int = 0
    cpu: int = 4
    batch: int = 64
    seed: int = 2023
    start_epoch: int = 0
    num_epoch: int = 100
    resume: str = 'false'
    
    # Model configurations
    backbone: str = 'inception'
    optim: str = 'adamw'
    task: Optional[str] = None
    adjust: Optional[str] = None
    meanvar: bool = False
    lr: float = 3e-4
    scheduler: str = 'off'
    
    # Loss weights
    lambda_ce: float = 1.0
    lambda_mean: float = 1.0
    lambda_var: float = 1.0

def parse_arguments(argv=None) -> BaseConfig:
    parser = argparse.ArgumentParser(description='Training configuration parser')
    
    # Hardware settings
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--seed', type=int, default=2023)
    
    # Training settings
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--resume', type=str, default='false', choices=['true', 'false'])
    
    # Model settings
    parser.add_argument('--backbone', type=str, default='inception', choices=['resnet34', 'inception'])
    parser.add_argument('--optim', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--task', type=str, required=True, choices=['fat', 'mmappend'])
    parser.add_argument('--adjust', type=str, required=True, choices=['base', 'a1', 'a2'])
    parser.add_argument('--meanvar', action='store_true',
                        help='Enable mean variance loss calculation')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--scheduler', type=str, default='off', choices=['on', 'off'])
    
    # Loss weights
    parser.add_argument('--lambda_ce', type=float, default=1.0)
    parser.add_argument('--lambda_mean', type=float, default=1.0)
    parser.add_argument('--lambda_var', type=float, default=1.0)
    
    args = parser.parse_args(argv)
    return BaseConfig(**vars(args))
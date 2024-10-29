from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
from .utils import TaskRanges

class LabelDistributionLearning:
    """Handles Label Distribution Learning calculations and losses"""
    TASK_RANGES = TaskRanges.RANGES

    @staticmethod
    def calculate_mae(inputs: Tensor, target: Tensor, task: str) -> Tensor:
        """Calculate mean average error for label distribution learning"""
        softmax = nn.Softmax(dim=1)
        pdf = softmax(inputs)
        
        if task not in LabelDistributionLearning.TASK_RANGES:
            raise ValueError(f"Unknown task: {task}")
            
        start, end, steps = LabelDistributionLearning.TASK_RANGES[task]
        class_list = torch.linspace(start, end, steps, device=inputs.device)
        
        mean = torch.squeeze((pdf * class_list).sum(1, keepdim=True))
        value = class_list[target]
        return torch.abs(mean - value).mean()

    @staticmethod
    def calculate_value(inputs: Tensor, task: str) -> Tensor:
        """Calculate prediction value using label distribution learning"""
        softmax = nn.Softmax(dim=1)
        pdf = softmax(inputs)
        
        if task not in LabelDistributionLearning.TASK_RANGES:
            raise ValueError(f"Unknown task: {task}")
            
        start, end, steps = LabelDistributionLearning.TASK_RANGES[task]
        class_list = torch.linspace(start, end, steps, device=inputs.device)
        
        distribution_value = torch.squeeze((pdf * class_list).sum(1, keepdim=True))
        return distribution_value.mean()


class MeanVarianceLoss(nn.Module):
    """Calculate mean and variance loss for label distribution learning"""
    def __init__(self, task: str):
        super().__init__()
        if task not in LabelDistributionLearning.TASK_RANGES:
            raise ValueError(f"Task {task} not supported. Available tasks: {list(LabelDistributionLearning.TASK_RANGES.keys())}")
        self.task = task
        self.softmax = nn.Softmax(dim=1)
        
    def _get_class_list(self) -> Tensor:
        start, end, steps = LabelDistributionLearning.TASK_RANGES[self.task]
        return torch.linspace(start, end, steps).to(self.get_device())
    
    def get_device(self) -> torch.device:
        if list(self.parameters()):
            return next(self.parameters()).device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, inputs: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        pdf = self.softmax(inputs)
        class_list = self._get_class_list()
        
        # Calculate mean loss
        mean = torch.sum(pdf * class_list, dim=1)
        target_values = class_list[target]
        mean_loss = torch.mean((mean - target_values) ** 2)
        
        # Calculate variance loss
        squared_diff = (class_list.unsqueeze(0) - mean.unsqueeze(1)) ** 2
        variance_loss = torch.mean(torch.sum(pdf * squared_diff, dim=1))
        
        return mean_loss, variance_loss
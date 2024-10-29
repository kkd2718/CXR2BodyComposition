from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from utils.utils import PathManager, RandomSeeder
from utils.ldl import LabelDistributionLearning
from utils.dataset import BodyCompositionInference
from utils.config_base import parse_arguments, BaseConfig

@dataclass
class ModelMetrics:
    predictions: Dict[str, Dict[str, float]]


class ModelInferencer:
    def __init__(self, config: BaseConfig, model: nn.Module, device: torch.device):
        self.config = config
        self.model = model
        self.device = device
        self.to_cpu = lambda x: x.detach().cpu()

    @torch.no_grad()
    def inference(self, data_loader: DataLoader) -> ModelMetrics:
        """Run inference on the dataset"""
        self.model.eval()
        predictions = defaultdict(dict)
        
        for _, data in enumerate(tqdm(data_loader, desc='Inference')):
            batch_predictions = self.process_batch(data)
            sample_id = data['ID'][0]
            
            for key, value in batch_predictions.items():
                predictions[key][sample_id] = value
                
        return ModelMetrics(predictions=dict(predictions))

    def process_batch(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        inputs = data['input'].to(self.device, dtype=torch.float)
        outputs = self.model(inputs)
        
        if self.config.backbone == 'inception':
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            
        if self.config.adjust == 'base':
            return self._process_base(outputs)
        elif self.config.adjust == 'a1':
            return self._process_a1(outputs)
        elif self.config.adjust == 'a2':
            return self._process_a2(outputs)
        else:
            raise ValueError(f"Unknown adjustment type: {self.config.adjust}")
        
    def _process_base(self, outputs: List[torch.Tensor]) -> Dict[str, float]:
        preds = {}
        preds[self.config.task] = LabelDistributionLearning.calculate_value(
            self.to_cpu(outputs[0]), self.config.task).item()
        return preds
        
    def _process_a1(self, outputs: List[torch.Tensor]) -> Dict[str, float]:
        preds = {}
        preds[self.config.task] = LabelDistributionLearning.calculate_value(
            self.to_cpu(outputs[0]), self.config.task).item()
        preds['age'] = LabelDistributionLearning.calculate_value(
            self.to_cpu(outputs[1]), 'age').item()
        _, sexes = torch.max(outputs[2], 1)
        preds['sex'] = sexes.item()
        return preds
        
    def _process_a2(self, outputs: List[torch.Tensor]) -> Dict[str, float]:
        preds = {}
        preds[self.config.task] = LabelDistributionLearning.calculate_value(
            self.to_cpu(outputs[0]), self.config.task).item()
        preds['age'] = LabelDistributionLearning.calculate_value(
            self.to_cpu(outputs[1]), 'age').item()
        _, sexes = torch.max(outputs[2], 1)
        preds['sex'] = sexes.item()
        preds['height'] = LabelDistributionLearning.calculate_value(
            self.to_cpu(outputs[3]), 'height').item()
        preds['weight'] = LabelDistributionLearning.calculate_value(
            self.to_cpu(outputs[4]), 'weight').item()
        return preds
    

class Inference:
    def __init__(self, config: BaseConfig):
        self.config = config
        self.device = self._setup_device()
        self._log_gpu_stats()

    def _setup_device(self) -> torch.device:
        device = torch.device(f'cuda:{self.config.gpu}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
        return device
    
    def _log_gpu_stats(self) -> None:
        print('Current GPU:', torch.cuda.current_device())
        if self.device.type == 'cuda':
            print('Allocated:', round(torch.cuda.memory_allocated(self.config.gpu)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(self.config.gpu)/1024**3,1), 'GB')
        print('Device:', self.device)

    def setup(self) -> None:
        RandomSeeder.seed_all(self.config.seed)
        print("Random seed fixed:", self.config.seed)
        self._setup_data()
        self._setup_model()

        self.inferencer = ModelInferencer(
            config=self.config,
            model=self.model,
            device=self.device
        )

    def _setup_data(self) -> None:
        dataset = BodyCompositionInference()
        self.data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )

    def _setup_model(self) -> None:
        """Setup model and load weights"""
        import model.inception as backbone
        self.model = backbone.Inception3(adjust=self.config.adjust, task=self.config.task)
        self.model.to(self.device)
        
        # Load checkpoint
        ckpt_path = self._get_checkpoint_path()
        checkpoint = torch.load(str(ckpt_path), map_location=f"cuda:{self.config.gpu}")
        self.model.load_state_dict(checkpoint['model'])
        
        print(f"Model setting completed: model - {self.config.backbone}")
        print(f"Total Parameter: {self.calculate_parameters(self.model):.2f}M")
        print(f"Trainable Parameter: {self.calculate_trainables(self.model):.2f}M")

    def _get_checkpoint_path(self) -> Path:
        ckpt_dir = PathManager.get_checkpoint_dir(self.config)
        ckpt_files = sorted(ckpt_dir.glob('*.pth'))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
        return ckpt_files[-1]
    
    @staticmethod
    def calculate_parameters(model: nn.Module) -> float:
        return sum(param.numel() for param in model.parameters())/1000000.0

    @staticmethod
    def calculate_trainables(model: nn.Module) -> float:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000.0
    
    def run(self) -> None:
        metrics = self.inferencer.inference(self.data_loader)
        self._save_results(metrics.predictions)
        
    def _save_results(self, predictions: Dict[str, Dict[str, float]]) -> None:
        csv_path = Path('inference_csv.csv')
        df = pd.read_csv(csv_path)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns='Unnamed: 0')
            
        # Add predictions to DataFrame
        base_name = f"{self.config.task}_{self.config.adjust}"
        df[base_name] = df['image_ID'].map(predictions[self.config.task])
        
        if self.config.adjust in ['a1', 'a2']:
            df[f"{base_name}_age"] = df['image_ID'].map(predictions['age'])
            df[f"{base_name}_sex"] = df['image_ID'].map(predictions['sex'])
            
        if self.config.adjust == 'a2':
            df[f"{base_name}_height"] = df['image_ID'].map(predictions['height'])
            df[f"{base_name}_weight"] = df['image_ID'].map(predictions['weight'])
            
        df.to_csv(csv_path, index=False)


def main():
    config = parse_arguments()
    inference = Inference(config)
    
    try:
        inference.setup()
        inference.run()
    except KeyboardInterrupt:
        print("\nInference interrupted by user")
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise


if __name__ == '__main__':
    main()
from pathlib import Path
from typing import Dict, List, Union
import pickle
import cv2
import numpy as np
import pandas as pd
import pydicom as dcm
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class BodyCompositionDevelop(Dataset):
    def __init__(self, mode: str):
        if mode not in ['train', 'tuning']:
            raise ValueError("Mode must be either 'train' or 'tuning'")
            
        self.mode = mode
        self.data = self._load_data()
        self.transform = self._get_transforms()
        
        self._load_attributes()

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        src_path = self._get_image_path(index)
        inputs = self._preprocessing(src_path)
        
        measurements = {
            'input': inputs,
            'age': self.age_lst[index],
            'sex': self.sex_lst[index],
            'height': self.height_lst[index],
            'weight': self.weight_lst[index],
            'fat': self.fat_lst[index],
            'mmwhole': self.muscle_whole_lst[index],
            'mmappend': self.muscle_append_lst[index],
            'wofat': self.without_fat_lst[index],
            'waist': self.waist_lst[index]
        }
        
        if self.transform:
            measurements['input'] = self.transform(image=inputs)['image']
            
        return measurements
    
    def _load_data(self) -> Dict:
        pkl_dir = Path('pkl_path')
        filename = 'train_set.pkl' if self.mode == 'train' else 'tuning_set.pkl'
        
        try:
            with open(pkl_dir / filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load {filename}: {str(e)}")
        
    def _load_attributes(self) -> None:
        self.image_paths = self.data['paths']
        self.age_lst = self.data['age']
        self.sex_lst = self.data['sex']
        self.height_lst = self.data['height']
        self.weight_lst = self.data['weight']
        self.fat_lst = self.data['fat']
        self.muscle_whole_lst = self.data['muscle_whole']
        self.muscle_append_lst = self.data['muscle_append']
        self.without_fat_lst = self.data['without_fat']
        self.waist_lst = self.data['waist']
    
    def _get_image_path(self, index: int) -> Path:
        return Path(self.image_paths[index][:-3] + 'png')
    
    @staticmethod
    def _preprocessing(src_path: Union[str, Path]) -> np.ndarray:
        img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {src_path}")
            
        img = img.astype('float32')
        img = img - img.min()
        img = img / (img.max() - img.min() + 1e-8)
        
        return img
        
    def _get_transforms(self) -> A.Compose:
        if self.mode == 'train':
            return A.Compose([
                A.ShiftScaleRotate(
                    scale_limit=(0.0, 0.1),
                    rotate_limit=15,
                    shift_limit=0.1,
                    p=1,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                A.GaussNoise(
                    var_limit=(0.0024902343750000003, 0.012451171875),
                    p=0.3
                ),
                A.Perspective(scale=(0.05, 0.1), p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.1, 0.1),
                        contrast_limit=(-0.2, 0.1),
                        p=1
                    ),
                    A.RandomGamma(gamma_limit=(60, 160), p=1),
                ], p=0.6),
                A.OneOf([
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                    A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.MotionBlur(blur_limit=3, p=1),
                    A.MedianBlur(blur_limit=3, p=0.1),
                ], p=0.3),
                ToTensorV2(),
            ])
        else:
            return A.Compose([ToTensorV2()])


class BodyCompositionInference(Dataset):
    def __init__(self):
        self.transform = A.Compose([ToTensorV2()])
        self._load_data()

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        try:
            src_path = Path(self.image_paths[index].replace('workspace', 'mnt/nas203'))
            inputs = self._preprocessing(src_path)
        except Exception as e:
            print(f"Warning: Failed to load image at index {index}: {str(e)}")
            inputs = np.zeros((512, 512))

        measurements = {
            'input': inputs,
            'age': self.age_lst[index],
            'sex': self.sex_lst[index],
            'height': self.height_lst[index],
            'weight': self.body_weight_lst[index],
            'ID': self.ID_list[index]
        }

        if self.transform:
            measurements['input'] = self.transform(image=inputs)['image']

        return measurements

    def _load_data(self) -> None:
        csv_path = Path('inference_csv.csv')
        try:
            df = pd.read_csv(csv_path)
            self.ID_list = list(df['image_ID'])
            self.image_paths = list(df['paths'])
            self.age_lst = list(df['age'])
            self.sex_lst = [0 if x == "F" else 1 for x in list(df['gender'])]
            self.height_lst = list(df['ht'])
            self.body_weight_lst = list(df['bwt'])
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV file: {str(e)}")
        
    @staticmethod
    def _preprocessing(src_path: Union[str, Path]) -> np.ndarray:
        """Preprocess DICOM image"""
        dcm_data = dcm.read_file(str(src_path), force=True)
        image = dcm_data.pixel_array

        if dcm_data[0x28, 0x4].value.lower() == "monochrome1":
            image = np.max(image) - image

        image = BodyCompositionInference._resize(512, image)
        image = image.astype('float32')
        image = image - image.min()
        image = image / (image.max() - image.min() + 1e-8)

        return image

    @staticmethod
    def _resize(image_size: int, im: np.ndarray) -> np.ndarray:
        """
        Resize image maintaining aspect ratio with zero padding
        
        Args:
            image_size: Target size (both width and height)
            im: Input image array
            
        Returns:
            Resized and padded image array
        """
        dtype = im.dtype
        res = np.zeros([image_size, image_size], dtype=dtype)
        
        # Convert to 2D if needed
        if len(im.shape) > 2:
            im = im[:, :, 0]
            
        im = im.astype(np.float)
        ori_h, ori_w = im.shape
        ori_L = max(ori_w, ori_h)
        scale = float(image_size) / ori_L
        
        # Resize maintaining aspect ratio
        new_size = (int(round(ori_w * scale)), int(round(ori_h * scale)))
        resized = cv2.resize(im, new_size, interpolation=cv2.INTER_LANCZOS4).astype(dtype)
        
        # Center crop/pad
        h, w = resized.shape[:2]
        if ori_w > ori_h:
            padding_size = (image_size - h) // 2
            res[padding_size:padding_size + h, :] = resized
        else:
            padding_size = (image_size - w) // 2
            res[:, padding_size:padding_size + w] = resized

        return res
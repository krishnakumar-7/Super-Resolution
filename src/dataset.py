import torch
import numpy as np
from torch.utils.data import Dataset

class TurbulenceDataset(Dataset):
    def __init__(self, data_path, mode='train', split_ratio=0.8):
        self.mode = mode
        print(f"[{mode.upper()}] Loading dataset from {data_path}...")
        
        # Load and split data
        full_data = np.load(data_path)
        split_idx = int(len(full_data) * split_ratio)
        
        if mode == 'train':
            self.data = full_data[:split_idx]
        else:
            self.data = full_data[split_idx:]
            
        self.mean = np.mean(self.data, axis=(0, 1, 2))
        self.std = np.std(self.data, axis=(0, 1, 2))
        
        print(f"[{mode.upper()}] Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        
        # Z-score normalization
        img = (img - self.mean) / (self.std + 1e-8)
        
        # Convert to Tensor
        tensor = torch.from_numpy(img).float().permute(2, 0, 1)
        return tensor

    def get_stats(self):
        return self.mean, self.std

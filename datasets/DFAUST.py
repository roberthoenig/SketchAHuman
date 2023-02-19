import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class DFAUST(Dataset):
    def __init__(self, latent_shapes_path, silhouettes_path):
        self.latent_shapes_path = latent_shapes_path
        self.silhouettes_path = silhouettes_path
        
        latent_shapes = np.load(latent_shapes_path)
        indices = list(range(0, 32928, 32))
        latent_shapes = latent_shapes[indices, :, :, :]
        latent_shapes = np.reshape(latent_shapes, (32928, 7, 9))
        latent_shapes = latent_shapes.reshape((32928, 63))
        latent_shapes = torch.Tensor(latent_shapes).float()
        self.latent_shapes = latent_shapes
        silhouettes =  np.load(self.silhouettes_path)
        silhouettes = torch.Tensor(silhouettes).float()
        silhouettes = silhouettes.unsqueeze(1)
        self.silhouettes = silhouettes

    def __len__(self):
        return self.latent_shapes.shape[0]

    def __getitem__(self, idx):
        sample = {'x': self.latent_shapes[idx],
                  'cond': self.silhouettes[idx]}
        return sample

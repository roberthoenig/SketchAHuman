import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class DFAUST(Dataset):
    def __init__(self, latent_shapes_path, silhouettes_path, train_test_ratio, n_samples, type):
        self.latent_shapes_path = latent_shapes_path
        self.silhouettes_path = silhouettes_path

        latent_shapes = np.load(self.latent_shapes_path)
        indices = list(range(0, 10, 32))
        latent_shapes = latent_shapes[indices, :, :, :]
        latent_shapes = np.reshape(latent_shapes, (32, 17, 9))
        latent_shapes = latent_shapes.reshape((32, 153))
        latent_shapes = latent_shapes*255
        latent_shapes = torch.Tensor(latent_shapes).float()
        self.latent_shapes = latent_shapes
        silhouettes = np.load(self.silhouettes_path)
        silhouettes = silhouettes[:32]
        silhouettes = torch.Tensor(silhouettes).float()
        silhouettes = silhouettes.unsqueeze(1)
        self.silhouettes = silhouettes
        if n_samples == -1:
            n_total_samples = self.latent_shapes.shape[0]
        else:
            n_total_samples = n_samples
        n_train_samples = int(n_total_samples * train_test_ratio)
        if type == 'train':
            self.silhouettes = self.silhouettes[:n_train_samples]
            self.latent_shapes = self.latent_shapes[:n_train_samples]
            self.n_samples = n_train_samples
        elif type == 'test':
            self.silhouettes = self.silhouettes[n_train_samples:]
            self.latent_shapes = self.latent_shapes[n_train_samples:]
            self.n_samples = n_total_samples - n_train_samples
        else:
            raise Exception(f"Unkown type {type}")
        print("Finished init, type=", type)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = {'x': self.latent_shapes[idx],
                  'cond': self.silhouettes[idx],
                  'idx': idx}
        return sample

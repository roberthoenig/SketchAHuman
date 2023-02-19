import torch
import numpy as np
from torch.utils.data import Dataset
import natsort
from os import listdir
from os.path import isfile, join
from PIL import Image

from utils.diffusion_utils import img_folder_to_np

class Silhouettes(Dataset):
    def __init__(self, dir, n_samples=None, sample_spacing=1):
        self.dir = dir
        self.n_samples = n_samples
        self.sample_spacing = sample_spacing
        fnames = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]
        self.fnames = natsort.natsorted(fnames, reverse=False)
        if n_samples is not None:
            self.fnames = self.fnames[:n_samples*sample_spacing:sample_spacing]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        with Image.open(self.dir + self.fnames[idx]) as img:
            img = img.resize((224, 224))
            arr = np.array(img)
            if len(arr.shape) == 3:
                arr = arr[:,:,0] == 255
            else:
                arr = arr == 255
        arr = torch.from_numpy(arr).float().unsqueeze(0)
        sample = {
            'name': self.fnames[idx],
            'cond': arr
        }
        return sample


            
                    

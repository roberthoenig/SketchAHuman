# sys.path.append(os.path.abspath('datasets'))
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from datasets.DFAUST import DFAUST

latent_shapes_path = "/home/robert/g/3DHumanGeneration/data/DFAUST/test_res.npy"
silhouettes_path = "/home/robert/g/3DHumanGeneration/data/DFAUST/test_res_silhouettes.npy"
train_test_ratio = 0.9
n_samples = -1

type = 'train'
dataset = DFAUST(latent_shapes_path, silhouettes_path, train_test_ratio, n_samples, type=type)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

for idx in range(0, len(dataset), 20):
    v = dataset[idx]['cond'].squeeze().numpy()
    im = Image.fromarray(np.uint8(v*255), 'L')
    im.save(type+"_samples/" + str(idx) + ".png")
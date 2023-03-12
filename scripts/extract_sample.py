# sys.path.append(os.path.abspath('datasets'))
import numpy as np
from PIL import Image
from datasets.DFAUST import DFAUST
import importlib
import torch
from plyfile import PlyData
from utils.diffusion_utils import ply_to_png
Param = importlib.import_module("submodules.3DHumanGeneration.code.GraphAE.utils.graphAE_param")
graphAE = importlib.import_module("submodules.3DHumanGeneration.code.GraphAE.Models.graphAE")
graphAE_dataloader = importlib.import_module("submodules.3DHumanGeneration.code.GraphAE.DataLoader.graphAE_dataloader")

latent_shapes_path = "/home/robert/g/3DHumanGeneration/data/DFAUST/test_res.npy"
silhouettes_path = "/home/robert/g/3DHumanGeneration/data/DFAUST/test_res_silhouettes.npy"
mesh_model_weight_path = "submodules/3DHumanGeneration/train/0422_graphAE_dfaust/weight_30/model_epoch0018.weight"
mesh_model_config_path = "submodules/3DHumanGeneration/train/0422_graphAE_dfaust/30_conv_pool.config"
train_test_ratio = 0.9
n_samples = -1

type = 'train'
idx = 23160
dataset = DFAUST(latent_shapes_path, silhouettes_path, train_test_ratio, n_samples, type=type)

# Mesh model
param_mesh = Param.Parameters()
param_mesh.read_config(mesh_model_config_path)
param_mesh.batch = 1
param_mesh.read_weight_path = mesh_model_weight_path
mesh_model = graphAE.Model(param_mesh, test_mode=True)
mesh_model.cuda()
checkpoint = torch.load(param_mesh.read_weight_path)
mesh_model.load_state_dict(checkpoint['model_state_dict'])
mesh_model.init_test_mode()
mesh_model.eval()
template_plydata = PlyData.read(param_mesh.template_ply_fn)

fname = str(idx)
# 3D mesh
mesh = dataset[idx]['x']
print("mesh.shape", mesh.shape)
mesh = mesh.reshape(1, 7, 9).cuda()
# mesh = torch.unsqueeze(mesh, dim=0).cuda()
print("mesh.shape", mesh.shape)
out_mesh = mesh_model.forward_from_layer_n(mesh, 8)
out_mesh = out_mesh.cpu()
pc_out = np.array(out_mesh[0].data.tolist())
graphAE_dataloader.save_pc_into_ply(template_plydata, pc_out, "individual_samples/" + (fname+".ply"))
ply_to_png("individual_samples/" + (fname+".ply"), "individual_samples/" + (fname + "_rendering" + ".png"), silhouette=False)

# Silhouette
v = dataset[idx]['cond'].squeeze().numpy()
im = Image.fromarray(np.uint8(v*255), 'L')
im.save("individual_samples/" + fname + ".png")
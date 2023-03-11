import numpy as np
from datasets.DFAUST import DFAUST
from datasets.Silhouettes import Silhouettes
import torch
import torch.optim as optim
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.EmbedImageCNN import EmbedImageCNN

from plyfile import PlyData
from utils.diffusion_utils import ply_to_png
# from submodels.3DHumanGeneration.Models import graphAE
import importlib

Param = importlib.import_module("submodules.3DHumanGeneration.code.GraphAE.utils.graphAE_param")
graphAE = importlib.import_module("submodules.3DHumanGeneration.code.GraphAE.Models.graphAE")
graphAE_dataloader = importlib.import_module("submodules.3DHumanGeneration.code.GraphAE.DataLoader.graphAE_dataloader")
from PIL import Image



class ShapeModelNoDiffusion():
    def __init__(self, config):
        self.config = config
        self.epoch = 1
        self.device = config["device"]
        self.model = None
        self.optimizer = None

    def save_model(self, appendix=''):
        path = self.config["experiment_dir"] / f'checkpoint_{self.epoch}_{appendix}.pt'
        self.model.cpu()
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        self.model.to(self.device)

    def parallel_to_cpu_state_dict(self, state_dict):
        deparalleled_state_dict = {}
        for k in state_dict.keys():
            s = "cond_model.module"
            new_k = k
            if k.startswith(s):
                new_k = "cond_model" + k[len(s):]
            deparalleled_state_dict[new_k] = state_dict[k]
        return deparalleled_state_dict

    def load_model(self, path):
        self.model.load_state_dict(self.parallel_to_cpu_state_dict(torch.load(path)['model_state_dict']))

    def train(self):
        # Dataset
        if self.config["dataset"] == "DFAUST":
            dataset = DFAUST(**self.config["dataset_args"], type='train')
            dataloader = DataLoader(dataset, batch_size=self.config["training"]["batch_sz"], shuffle=True,
                                    num_workers=0)
            dataset_eval = DFAUST(**self.config["dataset_args"], type='test')
            dataloader_eval = DataLoader(dataset_eval, batch_size=1, shuffle=False, num_workers=0)
        else:
            raise Exception(f'Unkown dataset {self.config["dataset"]}')
        print("Dataset loaded")
        # Model
        if self.config["Model"]["name"] == "EmbedImageCNN":
            self.model = EmbedImageCNN(**self.config["Model"]["args"])
        else:
            raise Exception(f'Unkown condition model {self.config["Model"]["name"]}')
        if "load_checkpoint" in self.config["Model"].keys():
            self.load_model(self.config["Model"]["load_checkpoint"])
        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        pbar = tqdm(range(self.config["training"]["n_epochs"]))
        loss_fn = torch.nn.MSELoss()
        for t in pbar:
            # Train
            losses = []
            for batch in dataloader:
                batch_x = batch['x'].to(self.device)
                cond_x = batch['cond'].float().to(self.device)
                y = self.model(cond_x)
                loss = loss_fn(batch_x, y)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()
                losses.append(loss.detach().item())
            batch_loss = np.array(losses).mean()
            # Eval
            losses_eval = []
            for batch in dataloader_eval:
                batch_x = batch['x'].to(self.device)
                cond_x = batch['cond'].float().to(self.device)
                with torch.no_grad():
                    y = self.model(cond_x)
                    loss = loss_fn(batch_x, y)
                losses_eval.append(loss.detach().item())
            eval_loss = np.array(losses_eval).mean()
            # Log
            pbar.set_postfix({'batch_loss': batch_loss, 'eval_loss': eval_loss})
            logging.info(f"Epoch {self.epoch}, average loss: {batch_loss}, average eval loss: {eval_loss}")
            if (t + 1) % self.config["training"]["epochs_per_checkpoint"] == 0:
                logging.info(f"Saving checkpoint.")
                self.save_model()
            self.epoch += 1

    def test(self):
        # Dataset
        if self.config["dataset"] == "Silhouettes":
            dataset = Silhouettes(**self.config["dataset_args"])
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        else:
            raise Exception(f'Unkown dataset {self.config["dataset"]}')

        # Model
        if self.config["Model"]["name"] == "EmbedImageCNN":
            self.model = EmbedImageCNN(**self.config["Model"]["args"])
        else:
            raise Exception(f'Unkown condition model {self.config["Model"]["name"]}')
        if "load_checkpoint" in self.config["Model"].keys():
            self.load_model(self.config["Model"]["load_checkpoint"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Mesh model
        param_mesh = Param.Parameters()
        param_mesh.read_config(self.config["MeshModel"]["config_path"])
        param_mesh.batch = 1
        param_mesh.read_weight_path = self.config["MeshModel"]["weight_path"]
        mesh_model = graphAE.Model(param_mesh, test_mode=True)
        mesh_model.cuda()
        checkpoint = torch.load(param_mesh.read_weight_path)
        mesh_model.load_state_dict(checkpoint['model_state_dict'])
        mesh_model.init_test_mode()
        mesh_model.eval()
        template_plydata = PlyData.read(param_mesh.template_ply_fn)

        for batch in dataloader:
            print("batch.keys", batch.keys())
            sample = self.model(batch['cond'])
            sample = np.concatenate([s.detach().numpy() for s in sample])
            sample = sample.reshape(-1, 17, 9)

            sample_torch = torch.FloatTensor(sample).cuda()

            mesh = sample_torch[-1]
            mesh = torch.unsqueeze(mesh, dim=0)
            out_mesh = mesh_model.forward_from_layer_n(mesh, 8)
            out_mesh = out_mesh.cpu()
            pc_out = np.array(out_mesh[0].data.tolist())
            # pc_out[:, 1] += 100
            fname = batch['name'][0]
            graphAE_dataloader.save_pc_into_ply(template_plydata, pc_out,
                                                str(self.config["experiment_dir"] / (fname + ".ply")))
            ply_to_png(str(self.config["experiment_dir"] / (fname + ".ply")),
                       str(self.config["experiment_dir"] / (fname + "_rendering" + ".png")), silhouette=False)
            v = batch['cond'].squeeze().numpy()
            im = Image.fromarray(np.uint8(v * 255), 'L')
            im.save(str(self.config["experiment_dir"] / (fname + ".png")))

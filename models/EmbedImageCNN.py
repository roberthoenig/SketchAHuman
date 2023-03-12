import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision

def preprocess_state_dict(state_dict):
    processed_state_dict = {}
    for k in state_dict.keys():
        s = "model."
        new_k = k
        if k.startswith(s):
            new_k = k[len(s):]
        processed_state_dict[new_k] = state_dict[k]
    return processed_state_dict

class EmbedImageCNN(nn.Module):
    def __init__(self, cond_sz, freeze_model, freeze_newlayers, load_checkpoint=None):
        super(EmbedImageCNN, self).__init__()
        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2))
        self.model.classifier = torch.nn.Sequential(
            torch.nn.AvgPool2d(13, 13),
            torch.nn.Flatten(),
            torch.nn.Linear(512, cond_sz),
        )
        self.model = torch.nn.Sequential(
            torchvision.transforms.Resize(224),
            torchvision.transforms.Pad(int((256 - 224) / 2)),
            self.model,
        )
        if load_checkpoint is not None:
            self.model.load_state_dict(preprocess_state_dict(torch.load(load_checkpoint)['model_state_dict']))
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        if freeze_newlayers:
            newlayers = [self.model[2].features[0], self.model[2].classifier]
            for layer in newlayers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, cond):
        out = self.model(cond)
        return out

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision

class EmbedImageCNN(nn.Module):
    def __init__(self, cond_sz, freeze):
        super(EmbedImageCNN, self).__init__()
        self.model = torchvision.models.squeezenet1_1(pretrained=True)
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
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel(self.model)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, cond):
        out = self.model(cond)
        return out

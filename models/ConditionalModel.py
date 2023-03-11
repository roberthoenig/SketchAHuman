import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import time


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.relu = nn.ReLU()
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, n_steps, in_sz, cond_sz, cond_model, do_cached_lookup):
        super(ConditionalModel, self).__init__()
        self.cond_model = cond_model
        self.lin1 = ConditionalLinear(in_sz+cond_sz, 256, n_steps)
        self.lin2 = ConditionalLinear(256, 256, n_steps)
        self.lin3 = ConditionalLinear(256, 256, n_steps)
        self.lin4 = nn.Linear(256, in_sz)
        self.cache = dict()
        self.idx_stores = set()
        self.do_cached_lookup = do_cached_lookup

    def cached_lookup(self, cond, idx):
        all_present = True
        for i in idx:
            if not i in self.idx_stores:
                all_present = False
                break
        if all_present == False:
            cond_out = self.cond_model(cond)
            for p, i in enumerate(idx):
                self.cache[i] = cond_out[p]
                self.idx_stores.add(i)
        cond_out = torch.stack([self.cache[i] for i in idx], dim=0)
        return cond_out

    def forward(self, x, y, cond, idx=None):
        if self.do_cached_lookup:
            cond_processed = self.cached_lookup(cond, [i.item() for i in idx])
        else:
            cond_processed = self.cond_model(cond)
        x = torch.cat([x, cond_processed], dim=-1)
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        x = self.lin4(x)
        return x

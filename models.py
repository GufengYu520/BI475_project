import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models.video

def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)

class R3DModel(nn.Module):
    def __init__(self, dropout):
        super(R3DModel, self).__init__()
        self.base_model = torchvision.models.video.r3d_18(pretrained=True)
        self.linear = nn.Linear(400, 3)

    def forward(self, vids):
        vids = vids.permute(0, 2, 1, 3, 4)
        x = self.base_model(vids)
        x = self.linear(x)
        return x





"""DeepONet surrogate — operator learning for EM parameter prediction."""
from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
class DeepONet(nn.Module):
    """Deep Operator Network for mapping geometry→EM response."""
    def __init__(self,branch_dim:int=100,trunk_dim:int=100,hidden:int=128):
        super().__init__()
        self.branch=nn.Sequential(nn.Linear(branch_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.trunk=nn.Sequential(nn.Linear(trunk_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,x_branch:torch.Tensor,x_trunk:torch.Tensor)->torch.Tensor:
        b=self.branch(x_branch);t=self.trunk(x_trunk)
        return torch.sum(b.unsqueeze(1)*t.unsqueeze(0),dim=-1)+self.bias
class DeepONetSurrogate:
    """DeepONet-based EM surrogate for frequency sweeps."""
    def __init__(self,geom_dim:int=256,freq_dim:int=1,hidden:int=128,device:str="cpu"):
        self.model=DeepONet(geom_dim,freq_dim,hidden).to(device);self.device=device
    def predict(self,geometry:torch.Tensor,frequencies:torch.Tensor)->torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            freq_input=frequencies.unsqueeze(-1).to(self.device)
            geom_input=geometry.unsqueeze(0).expand(len(frequencies),-1).to(self.device)
            return cast(torch.Tensor, self.model(geom_input, freq_input))

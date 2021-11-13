import torch
import torch.nn as nn


class OUFFN(nn.Module):
    def __init__(self, 
                 logdir,
                 lookback = 30,
                 random_seed=0, 
                 device = "cpu",
                 hidden_units = [4,4,4,4], 
                 dropout = 0.25):
        
        super(OUFFN, self).__init__()
        self.logdir = logdir
        self.random_seed = random_seed 
        torch.manual_seed(self.random_seed)
        self.device = torch.device(device)
        self.hidden_units = hidden_units
        self.is_trainable = True

        self.hiddenLayers = nn.ModuleList()
        for i in range(len(hidden_units)-1):
            self.hiddenLayers.append(nn.Sequential(
                nn.Linear(hidden_units[i], hidden_units[i+1]),
                # nn.ReLU(True),
                nn.Sigmoid(),
                nn.Dropout(dropout)))           
        self.finalLayer = nn.Linear(hidden_units[-1],1)
                
    def forward(self,x):
        for i in range(len(self.hidden_units)-1):
            x = self.hiddenLayers[i](x)
        return self.finalLayer(x).squeeze()

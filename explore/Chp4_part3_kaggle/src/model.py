import torch.nn as nn 

class Net(nn.Module):
    def __init__(self, in_feature:int):
        super().__init__()
        self.linear = nn.Linear(in_feature,1)
    
    def forward(self, x):
        return self.linear(x)
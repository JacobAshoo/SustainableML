import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.c1 = nn.Conv2d(3,32,3,padding=1)
    self.c2 = nn.Conv2d(32,64,3,padding=1)
    self.c3 = nn.Conv2d(64,128,3,padding=1)
    self.f  = nn.Linear(128,10)

  def forward(self,x):
    x = F.relu(self.c1(x))
    x = F.relu(self.c2(x))
    x = F.relu(self.c3(x))
    x = x.mean(dim=(2,3))
    return self.f(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Value(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(Value, self).__init__()
        input_dim: int = input_dim
        hidden_dim: int = 512
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) + x
        x = self.fc3(x)
        return x
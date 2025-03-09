from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(Policy, self).__init__()
        input_dim: int = input_dim
        hidden_dim: int = 64
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim + 1)  # all possible board positions + the NOOP action
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) + x
        x = self.fc3(x)
        return x

    def select_action(self, states: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Select an action given an observation and a mask of legal actions.
        
        Args:
            states (torch.Tensor): Tensor with shape (batch_size, board_size*board_size).
            masks (torch.Tensor): Binary tensor with shape (batch_size, board_size*board_size+1) indicating legal actions.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - action (torch.Tensor): Tensor of shape (batch_size, 2) with [row, col] pairs.
                - log_prob (torch.Tensor): Tensor of log probabilities for the sampled actions.
        """
        dist = self.dists(states, masks)
        return dist.sample()
    
    def log_probs(self, states: torch.Tensor, flat_actions: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, float]:
        assert(flat_actions.shape == (states.shape[0],1))
        dists = self.dists(states, masks)
        entropy = dists.entropy()
        return dists.log_prob(flat_actions.squeeze(-1)), entropy

    def dists(self, states, masks):
        logits = self.forward(states)
        probs = masked_softmax(logits, masks, dim=1)
        return torch.distributions.Categorical(probs=probs)

def masked_softmax(logits: torch.Tensor, masks: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply a masked softmax to the logits.
    
    Args:
        logits (torch.Tensor): The input logits.
        mask (torch.Tensor): A binary mask indicating valid entries.
        dim (int, optional): The dimension over which to apply softmax. Defaults to -1.
    
    Returns:
        torch.Tensor: The resulting probabilities after softmax.
    """
    masked_logits: torch.Tensor = logits.masked_fill(masks < 0.99999, float('-inf'))
    return F.softmax(masked_logits, dim=dim)

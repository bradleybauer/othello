import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(Policy, self).__init__()
        input_dim: int = input_dim
        hidden_dim: int = 1024
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim + 1)  # all possible board positions + the NOOP action
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) + x
        x = self.fc3(x)
        return x

    def select_action(self, observation: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Select an action given an observation and a mask of legal actions.
        
        Args:
            observation (torch.Tensor): Tensor with shape (batch_size, board_size*board_size).
            mask (torch.Tensor): Binary tensor with shape (batch_size, board_size*board_size+1) indicating legal actions.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - action (torch.Tensor): Tensor of shape (batch_size, 2) with [row, col] pairs.
                - log_prob (torch.Tensor): Tensor of log probabilities for the sampled actions.
        """
        logits = self.forward(observation)
        probs = masked_softmax(logits, mask, dim=1)
        dist = torch.distributions.Categorical(probs)
        flat_action = dist.sample()
        return flat_action
    
    def log_probs(self, observation: torch.Tensor, flat_action: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward(observation)
        probs = masked_softmax(logits, mask, dim=1)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(flat_action)

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Apply a masked softmax to the logits.
    
    Args:
        logits (torch.Tensor): The input logits.
        mask (torch.Tensor): A binary mask indicating valid entries.
        dim (int, optional): The dimension over which to apply softmax. Defaults to -1.
    
    Returns:
        torch.Tensor: The resulting probabilities after softmax.
    """
    masked_logits: torch.Tensor = logits.masked_fill(mask < 0.99999, float('-inf'))
    return F.softmax(masked_logits, dim=dim)

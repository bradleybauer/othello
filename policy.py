import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Policy(nn.Module):
    def __init__(self, board_size: int) -> None:
        super(Policy, self).__init__()
        self.board_size: int = board_size
        input_dim: int = board_size * board_size
        
        # Simple MLP architecture.
        self.fc1: nn.Linear = nn.Linear(input_dim, 128)
        self.fc2: nn.Linear = nn.Linear(128, 128)
        self.fc3: nn.Linear = nn.Linear(128, input_dim + 1)  # all possible board positions + the NOOP action
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            observation (torch.Tensor): Input tensor with shape (batch_size, board_size, board_size).
        
        Returns:
            torch.Tensor: The output logits.
        """
        # Flatten the board.
        x = observation.view(observation.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def select_action(self, observation: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select an action given an observation and a mask of legal actions.
        
        Args:
            observation (torch.Tensor): Tensor with shape (batch_size, board_size, board_size).
            mask (torch.Tensor): Binary tensor with shape (batch_size, board_size*board_size+1) indicating legal actions.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - action (torch.Tensor): Tensor of shape (batch_size, 2) with [row, col] pairs.
                - log_prob (torch.Tensor): Tensor of log probabilities for the sampled actions.
        """
        logits = self.forward(observation)

        # Apply masked softmax to obtain probabilities over actions.
        probs = masked_softmax(logits, mask, dim=1)

        # Create a categorical distribution and sample a flat action index.
        dist = torch.distributions.Categorical(probs)
        flat_action = dist.sample()  # Shape: (batch_size,)
        log_prob = dist.log_prob(flat_action)

        total_moves = self.board_size * self.board_size
        # Convert flat action to a 2D [row, col] pair.
        # For normal moves:
        #    row = action // board_size, col = action % board_size.
        # For NOOP (flat_action == total_moves):
        #    row = board_size (indicating NOOP), col = 0 (a canonical value).
        is_normal = flat_action < total_moves
        row: torch.Tensor = torch.where(
            is_normal,
            flat_action // self.board_size,
            torch.full_like(flat_action, self.board_size)
        )
        col: torch.Tensor = torch.where(
            is_normal,
            flat_action % self.board_size,
            torch.zeros_like(flat_action)
        )
        action = torch.stack([row, col], dim=1)

        return action, log_prob

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

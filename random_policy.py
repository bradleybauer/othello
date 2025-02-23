import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomPolicy(nn.Module):
    def __init__(self, board_size: int) -> None:
        super(RandomPolicy, self).__init__()
        self.input_dim: int = board_size * board_size

    def select_action(self, observation: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = torch.ones(1, self.input_dim + 1)
        probs = masked_softmax(logits, mask, dim=1)
        dist = torch.distributions.Categorical(probs)
        flat_action = dist.sample()
        return flat_action
    
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

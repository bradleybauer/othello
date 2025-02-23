import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomPolicy(nn.Module):
    def __init__(self):
        super(RandomPolicy, self).__init__()

    def select_action(self, observation: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid_choices = np.flatnonzero(mask.numpy())
        return torch.tensor([np.random.choice(valid_choices)])

        valid_choices = torch.where(mask > .5)[0]
        if valid_choices.numel() == 0:
            raise ValueError("No valid choices found.")
        random_index = torch.randint(0, valid_choices.numel(), (1,))
        return valid_choices[random_index]

        logits = torch.ones_like(mask)
        probs = masked_softmax(logits, mask, dim=1)
        dist = torch.distributions.Categorical(probs)
        flat_action = dist.sample()
        return flat_action
    
# def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
#     """
#     Apply a masked softmax to the logits.
#     Args:
#         logits (torch.Tensor): The input logits.
#         mask (torch.Tensor): A binary mask indicating valid entries.
#         dim (int, optional): The dimension over which to apply softmax. Defaults to -1.
#     Returns:
#         torch.Tensor: The resulting probabilities after softmax.
#     """
#     masked_logits: torch.Tensor = logits.masked_fill(mask < 0.99999, float('-inf'))
#     return F.softmax(masked_logits, dim=dim)

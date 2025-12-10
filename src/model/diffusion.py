"""
Discrete Diffusion model for applying noise to quantum gate sequences.
"""
import torch
import torch.nn.functional as F


class DiscreteDiffusion:
    """
    Simple discrete diffusion process for gate type corruption.
    Applies noise by randomly swapping gate types based on marginal distribution.
    """
    def __init__(self, marginal_list, T=100):
        """
        Args:
            marginal_list: List of tensors representing marginal distributions for each dimension
            T: Maximum timestep (controls noise intensity)
        """
        self.marginal_list = marginal_list
        self.T = T
        
    def apply_noise(self, x_0, t):
        """
        Apply discrete diffusion noise to input tensor.
        
        Args:
            x_0: Input tensor of shape (N, 1) with gate type indices
            t: Timestep tensor of shape (1,)
            
        Returns:
            tuple: (t, x_t) where x_t is the noisy version of x_0
        """
        t_val = t.item() if isinstance(t, torch.Tensor) else t
        intensity = min(t_val / self.T, 1.0)
        
        # Get marginal distribution (assuming single dimension for simplicity)
        marginal = self.marginal_list[0] if len(self.marginal_list) > 0 else torch.ones(1)
        num_classes = len(marginal)
        
        x_t = x_0.clone()
        batch_size = x_0.shape[0]
        
        # For each element, with probability = intensity, replace with random sample from marginal
        mask = torch.rand(batch_size, device=x_0.device) < intensity
        
        if mask.any():
            # Sample from marginal distribution
            probs = marginal / marginal.sum()
            noisy_indices = torch.multinomial(probs.unsqueeze(0), batch_size, replacement=True).squeeze(0)
            
            # Apply noise only to masked positions
            x_t[mask] = noisy_indices[mask].unsqueeze(1)
        
        return t, x_t


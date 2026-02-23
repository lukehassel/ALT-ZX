import torch
import torch.nn.functional as F


class DiscreteDiffusion:

    def __init__(self, marginal_list, T=100):

        self.marginal_list = marginal_list
        self.T = T
        
    def apply_noise(self, x_0, t):
        t_val = t.item() if isinstance(t, torch.Tensor) else t
        intensity = min(t_val / self.T, 1.0)
        
        marginal = self.marginal_list[0] if len(self.marginal_list) > 0 else torch.ones(1)
        num_classes = len(marginal)
        
        x_t = x_0.clone()
        batch_size = x_0.shape[0]
        
        mask = torch.rand(batch_size, device=x_0.device) < intensity
        
        if mask.any():
            probs = marginal / marginal.sum()
            noisy_indices = torch.multinomial(probs.unsqueeze(0), batch_size, replacement=True).squeeze(0)
            
            x_t[mask] = noisy_indices[mask].unsqueeze(1)
        
        return t, x_t


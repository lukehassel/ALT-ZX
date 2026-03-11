import torch
import torch.nn as nn
import torch.nn.functional as F


def L_odd(A, G_full, v_from):
    """
    Focused Odd Parity Loss:
    """
    n_measured = len(v_from)
    
    A_w = A[v_from, :].unsqueeze(0)
    
    G_u = G_full[v_from, :].unsqueeze(1)
    
    factors = 1.0 - 2.0 * A_w * G_u
    products = torch.prod(factors, dim=2)
    
    targets = 1.0 - 2.0 * torch.eye(n_measured, device=A.device)
    
    squared_errors = (products - targets) ** 2
    return torch.sum(squared_errors) / (4.0 * n_measured)


def L_order(G_full, tau, v_from, epsilon=0.1):
    """
    Order loss:
    """
    G_active = G_full[v_from, :]
    
    tau_u = tau[v_from].unsqueeze(1)
    tau_v = tau.unsqueeze(0)
    
    return torch.mean(G_active * F.relu(tau_u - tau_v + epsilon) ** 2)


def _expand_G(G, n, v_from, v_to, device):
    """
    Expand compact G matrix from sparse to full (nxn) matrix.
    """
    G_full = torch.zeros((n, n), device=device)
    rows = torch.tensor(v_from, device=device).unsqueeze(1).expand(-1, len(v_to))
    cols = torch.tensor(v_to, device=device).unsqueeze(0).expand(len(v_from), -1)
    G_full.index_put_((rows, cols), G)
    return G_full


class GFlowLoss(nn.Module):
    """
    Differentiable GFlow loss L(A).

    Optimizes over G and τ to find the global minimum
    """
    def __init__(self, n, inputs, outputs, inner_iterations=1000, lr=0.1):
        super().__init__()
        self.n = n
        self.inner_iterations = inner_iterations
        self.lr = lr
        self.v_from = sorted(set(range(n)) - set(outputs))
        self.v_to = sorted(set(range(n)) - set(inputs))
        self.tau_init = nn.Parameter(torch.rand(n))

    def forward(self, A):
        if len(self.v_from) == 0:
            return torch.tensor(0.0, device=A.device)

        # Find optimal G* and τ* with A detached
        A_det = A.detach()
        G_latent = nn.Parameter(torch.randn(len(self.v_from), len(self.v_to), device=A.device))
        tau = self.tau_init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([G_latent, tau], lr=self.lr)

        with torch.enable_grad():
            for _ in range(self.inner_iterations):
                opt.zero_grad()
                G = torch.sigmoid(G_latent)
                G_full = _expand_G(G, self.n, self.v_from, self.v_to, A.device)
                
                loss = L_odd(A_det, G_full, self.v_from) + L_order(G_full, tau, self.v_from)
                
                loss.backward()
                opt.step()
                if loss.item() < 0.001:
                    break

        G_star = torch.sigmoid(G_latent.detach())
        G_full_star = _expand_G(G_star, self.n, self.v_from, self.v_to, A.device)
        return L_odd(A, G_full_star, self.v_from) + L_order(G_full_star, tau.detach(), self.v_from)


if __name__ == "__main__":
    print("Testing GFlowLoss...")
    n = 3
    A_valid = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    model = GFlowLoss(n, [0], [2])
    print(f"Line graph loss:  {model(A_valid).item():.6f}")

    A_empty = torch.zeros((3, 3))
    print(f"No-edge loss:     {model(A_empty).item():.6f}")

import torch
import torch.nn as nn
import torch.nn.functional as F


def L_odd(A, G_full, v_from):
    """
    Odd loss
    """
    factors = 1 - 2 * A[v_from, :] * G_full[v_from, :]
    products = torch.prod(factors, dim=1)
    return torch.mean(0.25 * (products + 1) ** 2)


def L_order(G, tau, v_from, v_to, epsilon=0.1):
    """
    Order loss
    """
    tau_u = tau[v_from].unsqueeze(1)
    tau_v = tau[v_to].unsqueeze(0)
    return torch.mean(G * F.relu(tau_u - tau_v + epsilon) ** 2)


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
    Differentiable GFlow loss f(A).

    Optimizes over G and τ to find:
        f(A) = min_{G,τ} L(A, G, τ)
    """
    def __init__(self, n, inputs, outputs, inner_iterations=1000, lr=0.1):
        super().__init__()
        self.n = n
        self.inner_iterations = inner_iterations
        self.lr = lr
        self.v_from = sorted(set(range(n)) - set(outputs))  # Ō (measured)
        self.v_to = sorted(set(range(n)) - set(inputs))     # Ī (candidates)
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
                loss = L_odd(A_det, G_full, self.v_from) + 0.5 * L_order(G, tau, self.v_from, self.v_to)
                loss.backward()
                opt.step()
                if loss.item() < 0.001:
                    break

        # Re-evaluate with differentiable A
        G_star = torch.sigmoid(G_latent.detach())
        G_full_star = _expand_G(G_star, self.n, self.v_from, self.v_to, A.device)
        return L_odd(A, G_full_star, self.v_from) + 0.5 * L_order(G_star, tau.detach(), self.v_from, self.v_to)


if __name__ == "__main__":
    print("running ...")
    n = 3
    A = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    model = GFlowLoss(n, [0], [2])
    print(f"loss:  {model(A).item():.6f}")       # ≈ 0

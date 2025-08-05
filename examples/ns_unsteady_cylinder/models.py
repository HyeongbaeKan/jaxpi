"""PyTorch models for the unsteady cylinder example.

This file replaces the original JAX based implementation with a lightweight
PyTorch version.  The aim is not to provide a fully optimised solver but a
minimal model that mirrors the public API used by the training and evaluation
scripts.
"""

import torch
from torch import nn
from torch.autograd import grad


class NavierStokes2D(nn.Module):
    """Simple physics-informed network for the 2D Navier–Stokes equations."""

    def __init__(self, config, inflow_fn, temporal_dom, coords, Re):
        super().__init__()
        self.inflow_fn = inflow_fn
        self.temporal_dom = temporal_dom
        self.coords = coords
        self.Re = Re

        # Domain scaling factors
        self.L = coords[:, 0].max() - coords[:, 0].min()
        self.W = coords[:, 1].max() - coords[:, 1].min()

        # Build a simple feed forward network
        layers = []
        in_dim = 3
        hidden = config.arch.hidden_dim
        for _ in range(config.arch.num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 3))  # outputs: u, v, p
        self.network = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Helper networks
    # ------------------------------------------------------------------
    def neural_net(self, t, x, y):
        """Forward pass of the network with basic non-dimensionalisation."""
        t_n = t / self.temporal_dom[1]
        x_n = x / self.L
        y_n = y / self.W
        inputs = torch.stack([t_n, x_n, y_n], dim=-1)
        outputs = self.network(inputs)
        y_hat = y_n
        u = outputs[..., 0] + 4 * 1.5 * y_hat * (0.41 - y_hat) / (0.41**2)
        v = outputs[..., 1]
        p = outputs[..., 2]
        return u, v, p

    def forward(self, t, x, y):  # pragma: no cover - thin wrapper
        return self.neural_net(t, x, y)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    def w_net(self, t, x, y):
        """Compute vorticity ``w = dv/dx - du/dy``."""
        t.requires_grad_(True)
        x.requires_grad_(True)
        y.requires_grad_(True)
        u, v, _ = self.neural_net(t, x, y)
        u_y = grad(u, y, torch.ones_like(u), create_graph=True)[0]
        v_x = grad(v, x, torch.ones_like(v), create_graph=True)[0]
        return v_x - u_y

    def r_net(self, t, x, y):
        """Compute PDE residuals and outflow boundary conditions."""
        t.requires_grad_(True)
        x.requires_grad_(True)
        y.requires_grad_(True)
        u, v, p = self.neural_net(t, x, y)

        u_t = grad(u, t, torch.ones_like(u), create_graph=True)[0]
        v_t = grad(v, t, torch.ones_like(v), create_graph=True)[0]

        u_x = grad(u, x, torch.ones_like(u), create_graph=True)[0]
        v_x = grad(v, x, torch.ones_like(v), create_graph=True)[0]
        p_x = grad(p, x, torch.ones_like(p), create_graph=True)[0]

        u_y = grad(u, y, torch.ones_like(u), create_graph=True)[0]
        v_y = grad(v, y, torch.ones_like(v), create_graph=True)[0]
        p_y = grad(p, y, torch.ones_like(p), create_graph=True)[0]

        u_xx = grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        v_xx = grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]

        u_yy = grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
        v_yy = grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

        ru = u_t + u * u_x + v * u_y + p_x - (u_xx + u_yy) / self.Re
        rv = v_t + u * v_x + v * v_y + p_y - (v_xx + v_yy) / self.Re
        rc = u_x + v_y

        u_out = u_x / self.Re - p
        v_out = v_x

        return ru, rv, rc, u_out, v_out

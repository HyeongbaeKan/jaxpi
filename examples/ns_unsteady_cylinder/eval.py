"""Evaluation utilities for the unsteady cylinder example."""

from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import models
from utils import get_dataset, parabolic_inflow


def evaluate(config, workdir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        u_ref,
        v_ref,
        p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cylinder_coords,
        nu,
    ) = get_dataset()

    coords = coords.to(device)
    temporal_dom = torch.tensor([0.0, 1.0], device=device)
    inflow_fn = lambda y: parabolic_inflow(y, U_max=1.5)
    Re = 1.0 / nu.item() if nu.numel() == 1 else 1.0

    model = models.NavierStokes2D(config, inflow_fn, temporal_dom, coords, Re).to(device)
    ckpt = Path(workdir) / "ckpt" / config.wandb.name / "model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    with torch.no_grad():
        t = torch.tensor([temporal_dom[1]], device=device)
        u, v, p = model(t.repeat(coords.shape[0]), coords[:, 0], coords[:, 1])

    x = coords[:, 0].cpu().numpy()
    y = coords[:, 1].cpu().numpy()
    triang = tri.Triangulation(x, y)

    fig, axes = plt.subplots(3, 1, figsize=(6, 10))
    axes[0].tricontourf(triang, u.cpu().numpy(), levels=100, cmap="jet")
    axes[0].set_title("Predicted u")
    axes[1].tricontourf(triang, v.cpu().numpy(), levels=100, cmap="jet")
    axes[1].set_title("Predicted v")
    axes[2].tricontourf(triang, p.cpu().numpy(), levels=100, cmap="jet")
    axes[2].set_title("Predicted p")
    plt.tight_layout()

    save_dir = Path(workdir) / "figures" / config.wandb.name
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "prediction.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)

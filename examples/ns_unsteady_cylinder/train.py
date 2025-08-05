"""Training script for the unsteady cylinder example using PyTorch."""

from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import models
from utils import get_dataset, parabolic_inflow


def train_and_evaluate(config, workdir):
    """Train the physics informed network."""

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
    Re = 1.0 / nu.item() if nu.numel() == 1 else 1.0

    temporal_dom = torch.tensor([0.0, 1.0], device=device)
    inflow_fn = lambda y: parabolic_inflow(y, U_max=1.5)

    model = models.NavierStokes2D(config, inflow_fn, temporal_dom, coords, Re).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optim.learning_rate)

    writer = SummaryWriter(Path(workdir) / config.wandb.name)

    for step in range(config.training.max_steps):
        # Sample random residual points
        t = torch.rand(config.training.res_batch_size, device=device) * temporal_dom[1]
        idx = torch.randint(0, coords.shape[0], (config.training.res_batch_size,), device=device)
        x = coords[idx, 0]
        y = coords[idx, 1]

        ru, rv, rc, _, _ = model.r_net(t, x, y)
        loss = (ru.pow(2).mean() + rv.pow(2).mean() + rc.pow(2).mean())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.logging.log_every_steps == 0:
            writer.add_scalar("loss", loss.item(), step)
            print(f"step {step}: loss {loss.item():.6f}")

    writer.close()

    # Save final model
    save_dir = Path(workdir) / "ckpt" / config.wandb.name
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")

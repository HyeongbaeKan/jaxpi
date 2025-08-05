"""Utility helpers for the unsteady cylinder example.

This module previously relied on :mod:`jax.numpy` but the example has been
converted to run entirely with PyTorch.  We therefore replace usages of JAX
with either NumPy (for loading ``.npy`` files) or PyTorch tensors.
"""

import numpy as np
import torch


def parabolic_inflow(y, U_max):
    """Parabolic inflow profile used for boundary conditions.

    Args:
        y: Array of ``y`` coordinates.  Can be a NumPy array or a PyTorch
            tensor.  It is internally converted to a tensor for computation.
        U_max: Maximum velocity at the channel centreline.

    Returns:
        Tuple ``(u, v)`` of tensors representing the horizontal and vertical
        velocity components.
    """

    y_t = torch.as_tensor(y, dtype=torch.float32)
    u = 4 * U_max * y_t * (0.41 - y_t) / (0.41**2)
    v = torch.zeros_like(y_t)
    return u, v


def get_dataset():
    """Load the reference data used for training and evaluation.

    The original implementation returned JAX arrays.  Here we load the data
    using NumPy and convert the results to PyTorch tensors so they can be used
    directly in a PyTorch training loop.
    """

    data = np.load("data/ns_unsteady.npy", allow_pickle=True).item()
    u_ref = torch.from_numpy(data["u"]).float()
    v_ref = torch.from_numpy(data["v"]).float()
    p_ref = torch.from_numpy(data["p"]).float()
    t = torch.from_numpy(data["t"]).float()
    coords = torch.from_numpy(data["coords"]).float()
    inflow_coords = torch.from_numpy(data["inflow_coords"]).float()
    outflow_coords = torch.from_numpy(data["outflow_coords"]).float()
    wall_coords = torch.from_numpy(data["wall_coords"]).float()
    cylinder_coords = torch.from_numpy(data["cylinder_coords"]).float()
    nu = torch.as_tensor(data["nu"], dtype=torch.float32)

    return (
        u_ref,
        v_ref,
        p_ref,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        cylinder_coords,
        nu,
    )


def get_fine_mesh():
    """Load fine meshes used for evaluating the PDE residual."""

    data = np.load("data/fine_mesh.npy", allow_pickle=True).item()
    fine_coords = torch.from_numpy(data["coords"]).float()

    data = np.load("data/fine_mesh_near_cylinder.npy", allow_pickle=True).item()
    fine_coords_near_cyl = torch.from_numpy(data["coords"]).float()

    return fine_coords, fine_coords_near_cyl

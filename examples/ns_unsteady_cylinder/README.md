# Navier–Stokes flow around a cylinder

> **Note**: This example has been migrated from JAX to PyTorch.  The training
> and evaluation scripts in this directory now rely solely on PyTorch
> functionality.

## Usage

Open `cylinder_demo.ipynb` to run a short PyTorch training session and visualize predictions. This notebook replaces the previous `main.py` script.

## Problem Set-up

We follow the problem setup in [here](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html)




## Results

The model parameter can be found at [google drive link](https://drive.google.com/drive/folders/1wy_SJUMPOMFM19P9ChGu_cRlk99VRdZ1?usp=drive_link). For a comprehensive log of the loss and weights, please visit [our Weights & Biases dashboard](https://wandb.ai/jaxpi/ns_unsteady_cylinder?workspace=user-).


![ns_cylinder](figures/ns_cylinder_u.gif)

![ns_cylinder](figures/ns_cylinder_v.gif)

![ns_cylinder](figures/ns_cylinder_w.gif)

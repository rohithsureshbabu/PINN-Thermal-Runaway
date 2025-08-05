# 2D Transient Thermal PINN for Battery Cross-section with Internal Short Circuit (ISC)

This repository implements a Physics-Informed Neural Network (PINN) for solving 2D transient heat conduction in a layered battery cell geometry (electrode, separator, and collector), with a localized internal heat source due to an internal short circuit (ISC).

The network learns the solution to the heat equation:

$$
\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + \dot{q}_{\text{ISC}}
$$

using PyTorch's automatic differentiation, without requiring mesh generation or labeled data.

---

## Physics Setup

* **Domain**: 2D rectangular geometry (Length: 0.1 m, Height: 0.02 m)
* **Material Regions**:

  * **Collector** (bottom layer)
  * **Electrode** (middle layer)
  * **Separator** (top layer)
* **Thermal Properties**:

  * Spatially dependent anisotropic thermal conductivity `(kx, ky)` for each region
  * Constant volumetric heat capacity: $\rho c_p = 2 \times 10^6 \, \text{J/m}^3\text{K}$
* **Boundary and Initial Conditions**:

  * **Left boundary (x = 0)**: Dirichlet (fixed at T\_surface = 500 K)
  * **Right boundary (x = Lx)**: Convective heat loss to ambient (T\_inf = 298 K)
  * **Top and Bottom (y = 0, y = Ly)**: Neumann (insulated)
  * **Initial condition**: Uniform temperature T0 = 298 K
* **Internal Heat Source**:

  * Localized heat generation to simulate ISC (active only near `x < 0.005 m`)
  * Constant volumetric heat rate (default: 4e7 W/m³)

---

## PINN Architecture

* Input: spatial-temporal coordinates `(x, y, t)`
* Output: predicted temperature `T(x, y, t)`
* Network structure: fully connected MLP with `Sigmoid` activations
* Temperature is structured to automatically satisfy the left Dirichlet BC:

  $$
  T(x, y, t) = T_{\text{surface}} + x \cdot NN(x, y, t)
  $$

---

## Loss Functions

The total loss is composed of multiple physically motivated terms:

1. **Governing PDE residual**:

   * Evaluated via automatic differentiation
   * Includes spatial variation of thermal conductivity
2. **Initial condition loss**:

   * Penalizes deviation from T0 at $t = 0$
3. **Boundary condition loss**:

   * Includes convective and insulated boundaries
4. **Energy balance loss**:

   * Ensures net energy input results in appropriate temperature rise over time

Each term is scaled to improve training stability.

---

## Training Details

* Optimizer: Adam (`lr = 1e-3`)
* Epochs: 10,000
* Training data:

  * 2000 collocation points (PDE)
  * 500 points for BCs and initial condition
* Training is done on GPU if available, otherwise CPU

---

## Outputs

* Plots showing evolution of each loss term over epochs (PDE, IC, BC, energy)
* Temperature contour plot at a selected evaluation time (e.g., t = 30 s)
* (Optional) 1D temperature profile and 3D surface visualization

---

## File Structure

```
thermal_pinn/
├── main.py        # Contains model, training loop, and plotting
└── README.md      # Model description and usage
```

---

## Applications

* Modeling thermal effects during internal short circuits (e.g., nail penetration)
* Rapid thermal field estimation in battery cross-sections
* PINN benchmarking for layered and anisotropic domains
* Mesh-free, data-free PDE solving in battery systems

---

## Future Improvements

* Add support for electrochemical heat generation terms
* Include time-dependent or moving heat sources
* Export trained model for deployment or integration with other solvers
* Parameter sweep scripts for different ISC severity or geometries

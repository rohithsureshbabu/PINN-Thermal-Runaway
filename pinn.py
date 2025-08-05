# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt


# Physical and geometric parameters 
Lx, Ly       = 0.1, 0.02           
T0           = 298.0               
T_surface    = 500.0                
T_inf        = 298.0                
k_regions    = {  # Don't change                  
    'collector': (200, 200),         
    'separator': (0.5, 0.2),
    'electrode': (5, 3)
}
rho_cp       = 2e6               
h_conv       = 50.0              
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# PINN definition
class ThermalPINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        net = []
        for i in range(len(layers) - 2):
            net += [nn.Linear(layers[i], layers[i + 1]), nn.Sigmoid()] # Linear and Sigmoid
        net += [nn.Linear(layers[-2], layers[-1])] # Final linear layer for output
        self.net = nn.Sequential(*net)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y, t):
        inp = torch.cat([x, y, t], dim=1)
        raw_out = self.net(inp)
        return T_surface + x * raw_out



# Region selector: returns kx, ky based on (x,y)
def select_conductivity(x, y):
    kx = torch.zeros_like(x)
    ky = torch.zeros_like(x)
    # Custom collector, electrolyte and separator mask
    mask1 = (y < Ly / 10)
    mask2 = (y >= Ly / 10) & (y < 8 * Ly / 10)
    mask3 = (y >= 8 * Ly / 10)
    kx[mask1], ky[mask1] = k_regions['collector']
    kx[mask2], ky[mask2] = k_regions['electrode']
    kx[mask3], ky[mask3] = k_regions['separator']
    return kx, ky

def isc_heat_source(x):
    Q_val = 4e7  # Change as needed (W/m3)
    source_zone = x < 0.005   #Nail zone
    return Q_val * source_zone.float()

# Loss components
def loss_pde(model, x, y, t):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    T = model(x, y, t)
    kx, ky = select_conductivity(x, y)
    
    T_t = grad(T, t, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_x = grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_xx = grad(T_x, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_y = grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_yy = grad(T_y, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]

    q_dot = isc_heat_source(x)

# Loss function
    res = rho_cp * T_t - (kx * T_xx + grad(kx * T_x, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]) \
          - (ky * T_yy + grad(ky * T_y, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]) + q_dot
    return torch.mean(res ** 2)


def loss_ic(model, x, y):
    t0 = torch.zeros_like(x)
    return torch.mean((model(x, y, t0) - T0)**2)


def loss_bc(model, x_bc, y_bc, t_bc):
    # Ensure gradients can be computed wrt inputs
    x_bc = x_bc.clone().detach().requires_grad_()
    y_bc = y_bc.clone().detach().requires_grad_()
    t_bc = t_bc.clone().detach().requires_grad_()

    # Dirichlet: Left boundary: x = 0
    # already defined in the PINN class to concatenate T surface 
    # as T = Tsurface + x.NN

    # Right boundary: x = Lx, convective
    xR = Lx * torch.ones_like(y_bc)
    xR = xR.clone().detach().requires_grad_() 
    TR = model(xR, y_bc, t_bc)
    TR_x = torch.autograd.grad(TR, xR,
              grad_outputs=torch.ones_like(TR), create_graph=True)[0]
    kxR, _ = select_conductivity(xR, y_bc)
    bc2 = torch.mean(( -kxR * TR_x - h_conv*(TR - T_inf) )**2)

    # Bottom: y = 0, Neumann (insulated) 
    y0 = torch.zeros_like(x_bc)
    y0 = y0.clone().detach().requires_grad_()
    Tb = model(x_bc, y0, t_bc)
    Tb_y = torch.autograd.grad(Tb, y0,
              grad_outputs=torch.ones_like(Tb), create_graph=True)[0]
    bc3 = torch.mean(Tb_y**2)

    # Top: y = Ly, Neumann (insulated) 
    yT = Ly * torch.ones_like(x_bc)
    yT = yT.clone().detach().requires_grad_()
    Tt = model(x_bc, yT, t_bc)
    Tt_y = torch.autograd.grad(Tt, yT,
              grad_outputs=torch.ones_like(Tt), create_graph=True)[0]
    bc4 = torch.mean(Tt_y**2)

    return bc2 + bc3 + bc4


def loss_energy_balance(model, x, y, tf):
    t_final = tf * torch.ones_like(x)
    T_final = model(x, y, t_final)
    delta_T = T_final - T0
    energy = rho_cp * torch.mean(delta_T)
    return energy


# Training loop
model = ThermalPINN([3, 60, 60, 60, 1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_history = []
loss_pde_history = []
loss_ic_history = []
loss_bc_history = []
loss_energy_history = []
for epoch in range(10000):
    Nf, Nb = 2000, 500
    x_f = torch.rand(Nf, 1) * Lx
    y_f = torch.rand(Nf, 1) * Ly
    t_f = torch.rand(Nf, 1) * 1
    x_f, y_f, t_f = [v.to(device) for v in (x_f, y_f, t_f)]

    x_bc = torch.rand(Nb, 1) * Lx
    y_bc = torch.rand(Nb, 1) * Ly
    t_bc = torch.rand(Nb, 1) * 1
    x_bc, y_bc, t_bc = [v.to(device) for v in (x_bc, y_bc, t_bc)]

    L_pde = loss_pde(model, x_f, y_f, t_f)/1e13
    L_ic = loss_ic(model, torch.rand(Nb, 1).to(device) * Lx, torch.rand(Nb, 1).to(device) * Ly)/1e5
    L_bc = loss_bc(model, x_bc, y_bc, t_bc)/1e8
    L_energy = loss_energy_balance(model, x_f, y_f, t_f.max())/1e8

    loss = L_pde +  L_ic + L_bc + L_energy**2

    opt.zero_grad()
    loss.backward()
    opt.step()

    loss_history.append(loss.item())
    loss_pde_history.append(L_pde.item())
    loss_ic_history.append(L_ic.item())
    loss_bc_history.append(L_bc.item())
    loss_energy_history.append(L_energy.item())

    if epoch % 100 == 0:
        print(f"[{epoch}] Loss: {loss.item():.2e} | PDE: {L_pde:.2e} | IC: {L_ic:.2e} | BC: {L_bc:.2e} | Energy: {L_energy:.2e}")

epochs = range(len(loss_pde_history))

# PDE Residual Loss
plt.figure(figsize=(6,4))
plt.plot(epochs, loss_pde_history, 'b-')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('PDE Residual Loss (log scale)')
plt.title('PDE Residual Loss during Training')
plt.grid(True)
plt.tight_layout()
plt.show()

# Initial Condition Loss
plt.figure(figsize=(6,4))
plt.plot(epochs, loss_ic_history, 'g-')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Initial Condition Loss (log scale)')
plt.title('Initial Condition Loss during Training')
plt.grid(True)
plt.tight_layout()
plt.show()

# Boundary Condition Loss
plt.figure(figsize=(6,4))
plt.plot(epochs, loss_bc_history, 'r-')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Boundary Condition Loss (log scale)')
plt.title('Boundary Condition Loss during Training')
plt.grid(True)
plt.tight_layout()
plt.show()

# Energy Balance Loss
plt.figure(figsize=(6,4))
plt.plot(epochs, loss_energy_history, 'm-')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Energy Balance Loss (log scale)')
plt.title('Energy Balance Loss during Training')
plt.grid(True)
plt.tight_layout()
plt.show()
print("Training complete.")

# %%
# Evaluation time
t_eval = 30

# Grid resolution
nx, ny = 1000, 500
x = torch.linspace(0, Lx, nx)
y = torch.linspace(0, Ly, ny)
X, Y = torch.meshgrid(x, y, indexing='ij')  # Shape: (nx, ny)

# Flatten for model input
x_flat = X.reshape(-1, 1).to(device)
y_flat = Y.reshape(-1, 1).to(device)
t_flat = torch.full_like(x_flat, t_eval).to(device)

# Predict
model.eval()
with torch.no_grad():
    T_pred = model(x_flat, y_flat, t_flat)

# Reshape to grid
T_grid = T_pred.cpu().numpy().reshape(nx, ny)

plt.figure(figsize=(8, 4))
cp = plt.contourf(X.cpu(), Y.cpu(), T_grid, 50, cmap='hot_r')
plt.colorbar(cp, label='Temperature [K]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title(f'Temperature Distribution at t = {t_eval}s')
plt.tight_layout()
plt.show()


'''# Temperature along y at x = 0.001 m
x_val = 0.001
y_vals = torch.linspace(0, Ly, 100).reshape(-1, 1).to(device)
x_vals = torch.full_like(y_vals, x_val)
t_vals = torch.full_like(y_vals, t_eval)

with torch.no_grad():
    T_slice = model(x_vals, y_vals, t_vals).cpu().numpy()

plt.plot(y_vals.cpu().numpy(), T_slice)
plt.xlabel('y [m]')
plt.ylabel('T [K]')
plt.title(f'Temperature Profile at x = {x_val} m, t = {t_eval}s')
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X.cpu(), Y.cpu(), T_grid, cmap='hot')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('T [K]')
ax.set_title(f'Temperature Surface at t = {t_eval}s')
plt.tight_layout()
plt.show()
'''
# %%

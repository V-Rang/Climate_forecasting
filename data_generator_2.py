# Conv-Diff Equation:
# del(c)/del(t) = 0.05*[ -\nabla{\cdot}{\vec{v}c} + D\nabla^{2}{c} + |u|(1 - |u|^{2})]
# periodic BCs for top and bottom

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

# variables and discritization parameters

nt = 2001
nx = 64
ny = 64

dt = 5e-4 # lower to keep dissipation to later time-steps.

dx=2/(nx-1)
dy=2/(ny-1)

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.zeros((ny, nx)) #dtype = float64
v = np.zeros((ny, nx))

c_val = np.random.random((ny, nx))

c_overall = np.zeros((nt, ny, nx))

# velocity values = 0.1<-y, x>:
for j in range(ny):
  for i in range(nx):
    u[j,i] = -0.1 * j
    v[j,i] = 0.1 * i

X, Y = np.meshgrid(x, y)

diff_coeff = 1. # D

for n in range(nt):
  c_init = c_val.copy()
  for j in range(1, ny - 1):
    for i in range(1, nx - 1):
      adv_term = u[j,i] * (c_init[j,i] - c_init[j, i-1])/dx + v[j,i] * (c_init[j,i] - c_init[j -1, i]  )/dy
      diff_term = diff_coeff * (  ( c_init[j, i-1] - 2*c_init[j,i] + c_init[j, i+1] )/dx**2 + ( c_init[j-1, i] - 2*c_init[j,i] + c_init[j+1, i]   )/dy**2       )
      vel_magnitude = np.sqrt(u[j,i]**2 + v[j,i]**2)
      source_term = vel_magnitude * (1 - vel_magnitude**2)
      rhs_term = 0.05*(-adv_term + diff_term + source_term)
      c_val[j,i] = c_init[j,i] + dt * rhs_term
      c_overall[n, j, i] = c_val[j,i]
  
  # boundary conditions:
  # periodic:
  # top = bottom
  c_val[0,:] = c_val[-2,:]
  c_val[1,:] = c_val[-1,:]

  # left = right
  c_val[:,0] = c_val[:,-2]
  c_val[:,1] = c_val[:,-1]


def update_QOI(frame):
    ax.clear()
    contour = ax.contourf(X, Y, c_overall[frame], cmap="viridis")
    ax.set_title(f"2D evolution of QOI at time {frame*dt:.3f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return contour

fig, ax = plt.subplots(figsize=(6, 5))
contour = ax.contourf(X, Y, c_overall[0], cmap="viridis")
fig.colorbar(contour)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("2D Evolution of QOI")

# Create animation
ani = animation.FuncAnimation(fig, update_QOI, frames=c_overall.shape[0], interval=50, blit=False)
ani.save('evolution_QOI_1.gif', fps=20, writer='pillow')  # Requires pillow
np.save('QOI_1.npy', c_overall)
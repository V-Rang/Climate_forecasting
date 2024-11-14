import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# variables and discritization parameters

nt = 2001
nx = 64
ny = 64
c=1

n_random = 10  # Replace with the desired number of coordinates

# Generate random coordinates
# x_coords = np.random.randint(0, 64, n)
# y_coords = np.random.randint(0, 64, n)

# # Combine x and y coordinates into a list of tuples
# coordinates = list(zip(x_coords, y_coords))

nu = 1e-4
dt = 5e-4 # lower to keep dissipation to later time-steps.

dx=2/(nx-1)
dy=2/(ny-1)

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

comb = np.ones((ny, nx))
u = np.ones((ny, nx)) #dtype = float64
v = np.ones((ny, nx))
un = np.ones((ny, nx))
vn = np.ones((ny, nx))
uf = np.ones((nt,nx,ny))
vf = np.ones((nt,nx,ny))

# assigning initial conditions
u = np.random.random((ny, nx))
v = np.random.random((ny, nx))

X, Y = np.meshgrid(x, y)

# parameters for periodic bcs:
# source term
A_s = 10
kx_s = 5
ky_s = 10
omega_s = 0.2
sigma_s = 30

omega_noise = 0.5
num_points_source = 10
# bcs -> left inflow and right outflow:
u_0 = 4.

# moving center initial coordinates
# i0, j0 = 0.5, 0.5

##loop across number of time steps
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            # how to create a source term that injects dynamical activity:
            source_term = 0.
            #for a random set of points, every 20 timesteps, create a pulse:
            if( n % 20 == 0):
              x_coords = np.random.randint(0, nx, num_points_source)
              y_coords = np.random.randint(0, ny, num_points_source)
              coordinates = list(zip(x_coords, y_coords))
              if((i,j) in coordinates):
                vel_magnitude = np.sqrt(un[i,j]**2 + vn[i,j]**2)
                source_term = vel_magnitude*(1 - vel_magnitude**2)

            u_diss_term = (nu*dt/(dx**2))*(un[i+1,j]-2*un[i,j]+un[i-1,j])+(nu*dt/(dx**2))*(un[i,j-1]-2*un[i,j]+un[i,j+1])
            v_diss_term = (nu*dt/(dx**2))*(vn[i+1,j]-2*vn[i,j]+vn[i-1,j])+(nu*dt/(dx**2))*(vn[i,j-1]-2*vn[i,j]+vn[i,j+1])

            u[i,j] = (un[i, j] -(un[i, j] * dt / dx * (un[i, j] - un[i-1, j])) -vn[i, j] * dt / dy * (un[i, j] - un[i, j-1])) + u_diss_term + dt*source_term
            v[i,j] = (vn[i, j] -(un[i, j] * dt / dx * (vn[i, j] - vn[i-1, j]))-vn[i, j] * dt / dy * (vn[i, j] - vn[i, j-1])) + v_diss_term + dt*source_term
            uf[n,i,j] = u[i,j]
            vf[n, i, j] = v[i, j]

    # Velocity boundary conditions

    # instead of periodic, have inflow at left and outflow at right:
    u[:,0] = u_0 * np.sin(omega_s * n)  # left boundary = u(x = 0, y) = U_0 sin(wt).
    u[:,-1] = u[:,-2] # zero gradient at the outflow.

    u[0, :] = u[-2, :]  # Top boundary equals the second-to-last row
    u[-1, :] = u[1, :]  # Bottom boundary equals the second row

    v[0, :] = v[-2, :]  # Top boundary for v
    v[-1, :] = v[1, :]  # Bottom boundary for v


# Update function for animation
def update_u(frame):
    ax.clear()
    contour = ax.contourf(X, Y, uf[frame], cmap="viridis")
    ax.set_title(f"2D evolution of x-velocity at time {frame*dt:.3f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return contour

def update_v(frame):
    ax.clear()
    contour = ax.contourf(X, Y, vf[frame], cmap="viridis")
    ax.set_title(f"2D evolution of y-velocity at time {frame*dt:.3f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return contour

import matplotlib.animation as animation

# time evolution of uf:
fig, ax = plt.subplots(figsize=(6, 5))
contour = ax.contourf(X, Y, uf[0], cmap="viridis")
fig.colorbar(contour)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("2D Evolution of x-velocity")

# Create animation
ani = animation.FuncAnimation(fig, update_u, frames=uf.shape[0], interval=50, blit=False)
ani.save('evolution_x_velocity_compl22.gif', fps=20, writer='pillow')  # Requires pillow

np.save('conv_diff_u_compl22', uf)
np.save('conv_diff_v_compl22', vf)

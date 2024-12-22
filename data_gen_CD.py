# simulation of a conv-diff example in 2d without any source term (only convection and diffusion terms)
# Equation: dc/dt = c_{1}{ -nabla\cdot{\vec{v}c}} + c_{2}{-\nabla^{2}{c}}
# oscillating velocity field \vec{v} is created over the 2d domain [0,1] X [0,1], 48 by 48 grid points
# simulated over 500 t-steps, dt = 5e-3 sec.
# c is initialized to be the sum of 4 Gaussian peaks.
# c_1: convection coefficient = 1e-1
# c_2: diffusion coefficient = 1e-2



from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 1.0, 1.0  # Domain size
Nx, Ny = 48, 48  # Grid points
dx, dy = Lx /(Nx-1), Ly /(Ny - 1)  # Grid spacing

# Nt = 500
# dt = 5e-3

Nt = 999
dt = 25e-4

c1 = 1e-1 # high => more shock
c2 = 1e-2 # low => more shock

A = 1.0  # Velocity amplitude

# Discretized domain
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initial condition: Gaussian peak
# c = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.01)

r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)

# Initial condition: Two concentric Gaussian peaks
# A1, A2 = 1.0, 1.0      # Peak amplitudes
# r1, r2 = 0.2, 0.4      # Initial radii of the peaks
# sigma1, sigma2 = 0.02, 0.02  # Widths of the peaks
# c = A1 * np.exp(-(r - r1)**2 / sigma1**2) + A2 * np.exp(-(r - r2)**2 / sigma2**2)

sigma = 0.02  # Width of Gaussian peaks
peak1 = np.exp(-((X - 0.3)**2 + (Y - 0.3)**2) / sigma)
peak2 = np.exp(-((X - 0.7)**2 + (Y - 0.3)**2) / sigma)
peak3 = np.exp(-((X - 0.3)**2 + (Y - 0.7)**2) / sigma)
peak4 = np.exp(-((X - 0.7)**2 + (Y - 0.7)**2) / sigma)

# Combined initial condition
c = peak1 + peak2 + peak3 + peak4

# Helper function for Laplacian
def laplacian(f):
    return ( c[1:-1, 0:-2 ]  -2*c[1:-1, 1: -1] + c[1:-1, 2:] )/dx**2 + (  c[0:-2, 1:-1 ]  -2*c[1:-1, 1: -1] + c[2:, 1: -1]   )/dy**2

dc_conv_x = np.zeros((Ny, Nx))
dc_conv_y = np.zeros((Ny, Nx))
laplacian_c = np.zeros((Ny, Nx))

# Time integration
# n_steps = int(T / dt)
# snapshots = []
c_overall = np.zeros((Nt+1, Ny, Nx))
c_overall[0][1:-1, 1:-1] = c[1:-1, 1:-1]
omega = 2 * np.pi  # Frequency of oscillation

for n in range(Nt):
    for i in range(1, Nx-1):
      for j in range(1, Ny - 1):
          vr = np.sin(omega * (n+1)*dt)  
          vx = vr * (X - 0.5) / (r + 1e-12)  
          vy = vr * (Y - 0.5) / (r + 1e-12)

          # for dc/dx:
          if( i == 1):
            # second order forward
            # print(j, i)
            dc_conv_x[j, i] = (-3*c[j, i ] + 4 * c[j, i+1] - c[j, i+2])/(2*dx)
          elif(i == Nx-2):
            # second order backward
            dc_conv_x[j, i] = (3*c[j,i] - 4*c[j, i-1] + c[j, i-2])/ (2*dx)
          else:
            # fourth order central difference
            dc_conv_x[j, i] = ( -c[j, i+2] + 8 * c[j, i+1] - 8 * c[j, i -1] + c[j, i-2]  )/(12*dx)

          # for dc/dy:
          if(j == 1):
            # second order forward
            dc_conv_y[j,i] = (-3*c[j,i] + 4*c[j+1, i] - c[j+2, i])/(2*dy)
          elif(j == Ny-2):
            # second order backward
            dc_conv_y[j,i] = (3*c[j,i] - 4*c[j-1, i] + c[j-2, i])/(2*dy)
          else:
            # fourth order central difference
            dc_conv_y[j,i] = (-c[j+2, i] + 8*c[j+1, i] - 8*c[j-1, i] + c[j-2, i])/(12*dy)

          # laplacian:
          if( i == 1 or i == Nx - 2 or j == 1 or j == Ny-2):
            # second order central
            laplacian_c[j,i] = (c[j, i + 1] - 2*c[j,i] + c[j,i - 1])/dx**2 + (c[j+1, i] - 2*c[j,i] + c[j-1,i])/dy**2
          else:
            # fourth order central
            laplacian_c[j,i] = (-c[j, i+2] + 16*c[j, i+1] - 30*c[j,i] + 16*c[j,i-1] - c[j, i-2])/(12*dx**2) + \
            (-c[j+2, i] + 16*c[j+1, i] - 30*c[j,i] + 16*c[j-1,i] - c[j-2, i])/(12*dy**2)

    convection = c1 * (-vx[1:-1, 1:-1] * dc_conv_x[1:-1, 1:-1] - vy[1:-1,1:-1] * dc_conv_y[1:-1, 1:-1])

    # Diffusion term
    diffusion = c2 * laplacian_c[1:-1, 1:-1]

    # Update c using Forward Euler
    # c_overall[n+1][1:-1, 1:-1] = c[1:-1, 1:-1]
    if( n== 0):
      # c[1:-1, 1:-1] += dt * (convection + diffusion)
      c_overall[n+1][1:-1,1:-1] = c[1:-1,1:-1] + dt * (convection + diffusion)
    else:
      c_overall[n+1][1:-1,1:-1] = (1/3)* (4 * c_overall[n][1:-1, 1:-1] - c_overall[n-1][1:-1, 1:-1] + 2 * dt * (convection + diffusion) )

snapshots = c_overall.reshape(c_overall.shape[0], c_overall.shape[1] * c_overall.shape[2])
snapshots = snapshots.T
_, S, _ = np.linalg.svd(snapshots, full_matrices=False)

# # # Energy capture
cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
print(cumulative_energy[:10])
print("Number of modes for 95% energy:", np.searchsorted(cumulative_energy, 0.95) + 1)

# # make movie
test_arr = snapshots.T
np.save('snapshots_CD.npy', test_arr)
print(test_arr.shape)

test_arr = test_arr.reshape(test_arr.shape[0], Ny, Nx)
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(test_arr[0], cmap="viridis", interpolation="nearest")
plt.colorbar(im, ax=ax)

def update(frame):
    """Update the image for the given frame."""
    im.set_array(test_arr[frame])
    ax.set_title(f"Time Instant: {frame}")
    return [im]

# # Create the animation
ani = FuncAnimation(fig, update, frames=range(test_arr.shape[0]), interval=50, blit=True)
ani.save("movie_cd.mp4", writer="ffmpeg", fps=20)
plt.close(fig)

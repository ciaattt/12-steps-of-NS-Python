import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

def Poisson_Numerically_Live(dx, dy, p, B, convergence_res):
    
    l1norm = 1
    residuals = []
    iteration = 0

    # Setup real-time residual plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.semilogy([], [], label='Pressure Residual')
    ax.set_xlim(0, 10)
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual')
    ax.set_title('Real-Time Residual Convergence')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()

    while l1norm > convergence_res:
        p_new = p.copy()

        for i in range(1, p.shape[0]-1):
            for j in range(1, p.shape[1]-1):
                p_new[i, j] = (
                    ((p[i, j+1] + p[i, j-1]) * dy**2 +
                     (p[i+1, j] + p[i-1, j]) * dx**2 -
                     B[i, j] * dx**2 * dy**2)
                    / (2 * (dx**2 + dy**2))
                )

        # Dirichlet boundaries
        p_new[:, 0] = 0
        p_new[:, -1] = 0
        p_new[0, :] = 0
        p_new[-1, :] = 0

        # Compute residual
        l1norm = np.sum(np.abs(p_new - p)) / (np.sum(np.abs(p)) + 1e-10)
        residuals.append(l1norm)
        p = p_new
        iteration += 1

        # Update residual plot
        line.set_xdata(np.arange(len(residuals)))
        line.set_ydata(residuals)
        ax.set_xlim(0, len(residuals))
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

    plt.ioff()
    plt.show()

    return p_new, iteration

# ===================== Setup Domain ======================

# Grid size
nx, ny = 101, 51
a, b = 0, 2   # x-bound
c, d = 0, 1   # y-bound
dx = (b - a) / (nx - 1)
dy = (d - c) / (ny - 1)

x_vals = np.linspace(a, b, nx)
y_vals = np.linspace(c, d, ny)
X, Y = np.meshgrid(x_vals, y_vals)

# ===================== Initial Conditions ======================

# Pressure field
p = np.zeros((ny, nx))
p[:, -1] = y_vals  # P(x=2) = y

# Source term
B = np.zeros((ny, nx))
B[int(0.25/dy), int(0.5/dx)] = 100
B[int(0.75/dy), int(1.5/dx)] = -100

# ===================== Run Solver ======================

convergence_res = 1e-3
P_numeric, n_iter = Poisson_Numerically_Live(dx, dy, p, B, convergence_res)

# ===================== Plot Result ======================

fig1, ax1 = plt.subplots(figsize=(10, 7))
cax = ax1.imshow(P_numeric, cmap='seismic', origin='lower',
                 extent=[a, b, c, d],
                 vmin=np.min(P_numeric), vmax=np.max(P_numeric))
cbar = fig1.colorbar(cax, ax=ax1, orientation='horizontal')
cbar.set_label('Pressure $p$')
cbar.ax.xaxis.set_label_position('bottom')
cbar.ax.xaxis.tick_bottom()

ax1.set_title(f'2D Poisson Solution')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')

plt.show()
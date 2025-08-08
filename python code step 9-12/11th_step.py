import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from sympy import init_printing
from sympy.utilities.lambdify import lambdify
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
from matplotlib import cm  # colormap
init_printing(use_latex=True)

#set the element number for each direction square mesh
element_x = 101 #set for x element 
element_y = element_x #set for y element
a,b = 0,2   #set for x upper and lower bound
c,d = a,b   #set for y upper and lower bound
rho = 1     #set the density
nu = 0.01  #set the kinematic viscosity
t = 10
convergence_res = 0.001 #smaller means the source term diffuse better

#set the spacing for dx and dy
dx = (b-a)/(element_x-1)    #set for delta x
dy = (d-c)/(element_y-1)    #set for delta y
dt = 0.001

#set for x and y space respectively
x_vals = np.linspace(a,b,element_x)  #set for x
y_vals = np.linspace(c,d,element_y)  #set for y
X,Y = np.meshgrid(x_vals,y_vals)

#set initial condition, we perserve initial guessing
u_to = np.zeros([element_y,element_x]) #set the parameters u
v_to = np.zeros([element_y,element_x]) #set the parameters v
p_to = np.zeros([element_y,element_x]) #set the parameters p

def Pressure_Numerically(u_to,v_to,dx,dy,p,rho,dt,convergence_res):

    l1norm = 1  #initialize value
    convergence = []    #store residual

    while l1norm > convergence_res:

        p_new = p.copy()
        #we calculate the horizontal first then vertically declined after the horizontal is done
        for i in range (1,len(p[:,0])-1): #define the y each iteration

            for j in range (1,len(p[0])-1): #define the x each iteration

                #numeric poisson procedure
                p_new[i, j] = (
                    ((p[i, j+1] + p[i, j-1]) * dy**2 +
                     (p[i+1, j] + p[i-1, j]) * dx**2)
                    / (2 * (dx**2 + dy**2))
                    - (rho * dx**2 * dy**2) / (2 * (dx**2 + dy**2)) * (
                        (1/dt) * (
                            (u_to[i, j+1] - u_to[i, j-1]) / (2*dx) +
                            (v_to[i+1, j] - v_to[i-1, j]) / (2*dy)
                        )
                        - ((u_to[i, j+1] - u_to[i, j-1]) / (2*dx))**2
                        - 2 * (
                            (u_to[i+1, j] - u_to[i-1, j]) / (2*dy) *
                            (v_to[i, j+1] - v_to[i, j-1]) / (2*dx)
                        )
                        - ((v_to[i+1, j] - v_to[i-1, j]) / (2*dy))**2
                    )
                )
            
        p_new[:,0] = p_new[:,1]      #set the x = 0 ,dP = 0 neumann boundary
        p_new[:,-1] = p_new[:,-2]       #set the x = 2 ,dP = 0 neumann boundary
        p_new[0, :] = p_new[1,:]  #set the y = 0 , dP = 0 neumann boundary
        p_new[-1,:] = 0 #set the y = 2 dirichlet boundary

        #calculate the error and pass to next calculation to reduce error
        l1norm = np.abs(np.sum(np.abs(p_new[:]) - np.abs(p[:]))/(np.sum(np.abs(p[:]))) + 1e-10) 
        print(f'convergence residual: {l1norm}')
        p = p_new

    return p_new

#def apply_boundary_conditions(p):
#    # Neumann BC
#    p[:, 0]  = p[:, 1]    # dp/dx = 0 at x = 0
#    p[:, -1] = p[:, -2]   # dp/dx = 0 at x = Lx
#    p[0, :]  = p[1, :]    # dp/dy = 0 at y = 0
#    
#    # Dirichlet BC
#    p[-1, :] = 0          # p = 0 at y = Ly
#    return p
#
#def Pressure_Numerically(u, v, dx, dy, p, rho, dt, convergence_res):
#    l1norm = 1.0  # Initial residual
#    
#    while l1norm > convergence_res:
#        p_new = p.copy()
#        
#        for i in range(1, p.shape[0] - 1):   # Loop over y
#            for j in range(1, p.shape[1] - 1): # Loop over x
#                
#                # First derivatives
#                du_dx = (u[i, j+1] - u[i, j-1]) / (2 * dx)
#                du_dy = (u[i+1, j] - u[i-1, j]) / (2 * dy)
#                dv_dx = (v[i, j+1] - v[i, j-1]) / (2 * dx)
#                dv_dy = (v[i+1, j] - v[i-1, j]) / (2 * dy)
#                
#                # Poisson equation (Pressure from continuity + momentum terms)
#                p_new[i, j] = (
#                    ((p[i, j+1] + p[i, j-1]) * dy**2 +
#                     (p[i+1, j] + p[i-1, j]) * dx**2)
#                    / (2 * (dx**2 + dy**2))
#                    - rho * dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * (
#                        (1/dt) * (du_dx + dv_dy)
#                        - du_dx**2
#                        - 2 * du_dy * dv_dx
#                        - dv_dy**2
#                    )
#                )
#        
#        # Apply boundary conditions
#        p_new = apply_boundary_conditions(p_new)
#        
#        # Compute residual
#        l1norm = np.sum(np.abs(p_new - p)) / np.sum(np.abs(p) + 10e-6)
#        print(f'Convergence residual: {l1norm:.6e}')
#        
#        # Update pressure field
#        p = p_new
#    
#    return p_new

#construct the numerical framework of 2d convection equation
def Momentum_Numerically(u_to,v_to,p_to,dx,dy,dt,nu,rho,convergence_res):

    U = []  #tensor to save all the u data spaciotemporal
    V = []  #tensor to save all the v data spaciotemporal

    for k in range (0,int(t/dt)):

        print (f'timestep : {k}')
        u_new = u_to.copy() #copy to get identic matrices then by next line we manipulate each column and row by numeric procedure
        v_new = v_to.copy() #copy to get identic matrices then by next line we manipulate each column and row by numeric procedure

        p_new = Pressure_Numerically(u_to,v_to,dx,dy,p_to,rho,dt,convergence_res) #set the pressure distribution

        #we calculate the horizontal first then vertically declined after the horizontal is done
        #numeric burger procedure
        for i in range (1,len(u_to[:,0])-1): #define the y each iteration

            for j in range (1,len(u_to[0])-1): #define the x each iteration
                u_new[i, j] = (
                    u_to[i, j]
                    - dt * u_to[i, j] * (u_to[i, j] - u_to[i, j-1]) / dx   # ∂u/∂x
                    - dt * v_to[i, j] * (u_to[i, j] - u_to[i-1, j]) / dy   # ∂u/∂y
                    - dt / (rho * 2*dx) * (p_new[i, j+1] - p_new[i, j-1])  # ∂p/∂x
                    + nu * dt * (
                        (u_to[i, j+1] - 2*u_to[i, j] + u_to[i, j-1]) / dx**2
                        + (u_to[i+1, j] - 2*u_to[i, j] + u_to[i-1, j]) / dy**2
                    )
                )

                v_new[i, j] = (
                    v_to[i, j]
                    - dt * u_to[i, j] * (v_to[i, j] - v_to[i, j-1]) / dx   # ∂v/∂x
                    - dt * v_to[i, j] * (v_to[i, j] - v_to[i-1, j]) / dy   # ∂v/∂y
                    - dt / (rho * 2*dy) * (p_new[i+1, j] - p_new[i-1, j])  # ∂p/∂y
                    + nu * dt * (
                        (v_to[i, j+1] - 2*v_to[i, j] + v_to[i, j-1]) / dx**2
                        + (v_to[i+1, j] - 2*v_to[i, j] + v_to[i-1, j]) / dy**2
                    )
                )
                
                #c_u = u_new[i,j]*dt/dx
                #c_v = v_new[i,j]*dt/dy
                #print (f'courant number: {np.mean([c_u,c_v])}')

        #set the boundary condition
        u_new[0,:] = 0        #at y = 0 u = 0
        u_new[-1,:] = 1       #at y = 2 u = 1
        u_new[:,0] = 0      #at x = 0 u = 0
        u_new[:,-1] = 0     #at x = 2 u = 0     
        
        v_new[0,:] = 0        #at y = 0 v = 0
        v_new[-1,:] = 0       #at y = 2 v = 0
        v_new[:,0] = 0      #at x = 0 v = 0
        v_new[:,-1] = 0     #at x = 2 v = 0

        cfl_u = np.max(np.abs(u_new)) * dt / dx
        cfl_v = np.max(np.abs(v_new)) * dt / dy
        cfl = max(cfl_u, cfl_v)
        print(f"Max CFL: {cfl:.3e}")

        #store each timestep iteration
        U.append(u_new) 
        V.append(v_new)
        #input the u after each timestep and refresh to next timestep
        u_to = u_new
        v_to = v_new
        p_to = p_new

    return U,V

U_xyt = Momentum_Numerically(u_to,v_to,p_to,dx,dy,dt,nu,rho,convergence_res)[0]

fig, ax = plt.subplots(figsize=(8,7))

# Inisialisasi heatmap awal
cax = ax.imshow(U_xyt[0], cmap='jet', origin='lower', extent=[0, 2, 0, 2], vmin=np.min(U_xyt), vmax=np.max(U_xyt))
fig.colorbar(cax, ax=ax)
ax.set_title('2D NS Equation: $u(x, y, t)$')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

def animate(n):
    cax.set_data(U_xyt[n])
    ax.set_title(f'2D Burger Equation $u(x, y, t)$ at t = {n}')
    return cax,

ani = animation.FuncAnimation(fig, animate, frames=len(U_xyt), interval=30, blit=False)
#ani.save('savedata/burger_2d_animation U.mp4', writer='ffmpeg', fps=30)

plt.tight_layout()
plt.show()



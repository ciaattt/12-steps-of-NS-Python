import numpy as np
import pandas as pd
import sympy 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from sympy import init_printing
from sympy.utilities.lambdify import lambdify
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
from matplotlib import cm  # colormap
init_printing(use_latex=True)

#set several domain data discrete
element_num_x = 101 #we set 101
element_num_y = 51 #we set 51
a,b = 0,2 #set the x bound
c,d = 0,1 #set the y bound
dx = (b-a)/(element_num_x-1)  #set the dx length
dy = (d-c)/((element_num_y-1))  #set the dy length
convergence_res = 0.0001 #smaller means the source term diffuse better

#mesh to visualize and numerically matter
x_vals = np.linspace(a,b,element_num_x) #set the x axis
y_vals = np.linspace(c,d,element_num_y) #set the y axis
X, Y = np.meshgrid(x_vals, y_vals) #set for meshgrid

#set for boundary condition
p = np.zeros([element_num_y,element_num_x])
p[:,0] = 0      #set the x = 0 ,P = 0
p[:,100] = y_vals       #set the x = 2 ,P = y

#set for nonhomogenouos (source) term
B = np.zeros([element_num_y,element_num_x])
B[int(0.25/dy),int(0.5/dx)] = 100
B[int(0.75/dy),int(1.5/dx)] = -100

def Poisson_Numerically(dx,dy,p,B,convergence_res):

    l1norm = 1  #initialize value
    convergence = []    #store residual

    while l1norm > convergence_res:

        p_new = p.copy()
        #we calculate the horizontal first then vertically declined after the horizontal is done
        
        for i in range (1,len(p[:,0])-1): #define the y each iteration

            for j in range (1,len(p[0])-1): #define the x each iteration

                #numeric laplace procedure
                p_new[i,j] = (
                    ((p[i,j+1] + p[i,j-1]) * dy**2 +
                     (p[i+1,j] + p[i-1,j]) * dx**2 - B[i,j] * dx**2 * dy**2)
                    / (2 * (dx**2 + dy**2))
                )
            
        p_new[:,0] = 0      #set the x = 0 ,P = 0 dirichlet
        p_new[:,-1] = 0       #set the x = 2 ,P = 0 dirichlet
        p_new[0,  :] = 0  #set the dirichlet boundary
        p_new[-1, :] = 0 #set the dirichlet boundary

        #calculate the error and pass to next calculation to reduce error
        l1norm = np.abs(np.sum(np.abs(p_new[:]) - np.abs(p[:]))/(np.sum(np.abs(p[:])))) 
        print(f'convergence residual: {l1norm}')
        convergence.append(l1norm)
        p = p_new

    # Plot residual convergence
    plt.figure(figsize=(8, 5))
    plt.semilogy(convergence, label='L1 Norm (Residual)')
    plt.xlabel('Iteration')
    plt.ylabel('Residual (log scale)')
    plt.title('Convergence of Poisson Equation Solver')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return p_new

fig, ax = plt.subplots(figsize=(10,7))  #initialize the figure by subplots

P_numeric = Poisson_Numerically(dx,dy,p,B,convergence_res)

# Inisialisasi heatmap awal
cax = ax.imshow(P_numeric, cmap='seismic', origin='lower', extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)], vmin=np.min(P_numeric), vmax=np.max(P_numeric)) #extent to visualize 
cbar = fig.colorbar(cax, ax=ax, orientation = 'horizontal')
cbar.ax.xaxis.set_label_position('bottom')
cbar.ax.xaxis.tick_bottom()  # Put ticks on top
ax.set_title('2D Numeric Poisson Equation: $u(x, y)$')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

plt.show()
#res_1,res_2 = Laplace_Analytically(X,Y)
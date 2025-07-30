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

element_num = 101 #we set 101, and make the mesh with square (quadra shape)
a,b = 0,2 #set the x bound
c,d = 0,2 #set the y bound
dx = (b-a)/(element_num-1)  #set the dx length
dy = (d-c)/(element_num-1)  #set the dy length
t = 0.625   #set the t limit
dt = 0.001  #set the timestep

x_vals = np.linspace(a,b,element_num) #set the x axis
y_vals = np.linspace(c,d,element_num) #set the y axis
X, Y = np.meshgrid(x_vals, y_vals) #set for meshgrid

#set the initial condition, at u(x,y,0)
u = np.ones((element_num,element_num))  # u = 1 everywhere except below
u[int(0.5/dx):int(1/dx),int(0.5/dy):int(1/dy)] = 2 # 0.5 <= (x,y) <=1, u = 2
u_to = u #just redefine this is at t=0

#set the initial condition, at v(x,y,0)
v = np.ones((element_num,element_num))  # u = 1 everywhere except below
v[int(0.5/dx):int(1/dx),int(0.5/dy):int(1/dy)] = 2 # 0.5 <= (x,y) <=1, u = 2
v_to = v #just redefine this is at t=0

# Plotting 3D surface of Boundary Condition
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, u_to, cmap=cm.viridis, edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('u(x, y, 0)')
ax.set_title('3D Surface Plot of u(x, y, 0) 2D Linear Convection')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.tight_layout()
plt.show()

#construct the numerical framework of 2d convection equation
def Convection_2D(u_to,v_to,dx,dy,dt):

    U = []  #tensor to save all the u data spaciotemporal
    V = []  #tensor to save all the v data spaciotemporal

    for i in range (0,int(t/dt)):

        u_new = u_to.copy() #copy to get identic matrices then by next line we manipulate each column and row by numeric procedure
        v_new = v_to.copy() #copy to get identic matrices then by next line we manipulate each column and row by numeric procedure

        #we calculate the horizontal first then vertically declined after the horizontal is done
        for i in range (0,len(u_to[:,0])): #define the y each iteration

            for j in range (0,len(u_to[0])): #define the x each iteration
                #numeric convection procedure
                u_new[i,j] = (u_to[i,j] 
                              - v_to[i,j]*(dt/dy)*(u_to[i,j] - u_to[i-1,j]) 
                              - u_to[i,j]*(dt/dx)*(u_to[i,j] - u_to[i,j-1]))

                v_new[i,j] = (v_to[i,j] 
                              - v_to[i,j]*(dt/dy)*(v_to[i,j] - v_to[i-1,j]) 
                              - u_to[i,j]*(dt/dx)*(v_to[i,j] - v_to[i,j-1]))
                
        #set the boundary condition
        u_new[0] = 1        #at x = 0 u = 1
        u_new[-1] = 1       #at x = 2 u = 1
        u_new[:,0] = 1      #at y = 0 u = 1
        u_new[:,-1] = 1     #at y = 2 u = 1     
        
        v_new[0] = 1        #at x = 0 v = 1
        v_new[-1] = 1       #at x = 2 v = 1
        v_new[:,0] = 1      #at y = 0 v = 1
        v_new[:,-1] = 1     #at y = 2 v = 1

        #store each timestep iteration
        U.append(u_new) 
        V.append(v_new)
        #input the u after each timestep and refresh to next timestep
        u_to = u_new
        v_to = v_new

    return U,V

#call the U
U_xyt = Convection_2D(u_to,v_to,dx,dy,dt)[0]

#call the U
U_xyt = Convection_2D(u_to,v_to,dx,dy,dt)[0]
V_xyt = Convection_2D(u_to,v_to,dx,dy,dt)[1]

#Visualize U
fig, ax = plt.subplots(figsize=(8,7))

# Inisialisasi heatmap awal
cax = ax.imshow(U_xyt[0], cmap='magma', origin='lower', extent=[0, 2, 0, 2], vmin=np.min(U_xyt), vmax=np.max(U_xyt))
fig.colorbar(cax, ax=ax)
ax.set_title('2D Scalar Field: $u(x, y, t)$')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

def animate(n):
    cax.set_data(U_xyt[n])
    ax.set_title(f'2D Field $u(x, y, t)$ at t = {n}')
    return cax,

ani = animation.FuncAnimation(fig, animate, frames=len(U_xyt), interval=30, blit=False)

plt.tight_layout()
plt.show()

#Visualize V
fig, bx = plt.subplots(figsize=(8,7))

# Inisialisasi heatmap awal
dax = bx.imshow(V_xyt[0], cmap='magma', origin='lower', extent=[0, 2, 0, 2], vmin=np.min(V_xyt), vmax=np.max(V_xyt))
fig.colorbar(dax, ax=bx)
bx.set_title('2D Scalar Field: $u(x, y, t)$')
bx.set_xlabel('$x$')
bx.set_ylabel('$y$')

def animate(n):
    dax.set_data(V_xyt[n])
    bx.set_title(f'2D Field $v(x, y, t)$ at t = {n}')
    return dax,

ani = animation.FuncAnimation(fig, animate, frames=len(V_xyt), interval=30, blit=False)

plt.tight_layout()
plt.show()
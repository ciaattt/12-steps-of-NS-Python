import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

element_num = 100   #element number
u = np.ones(element_num)    #meshing resolution
a,b = 0,2   #set the lower and upper bound respectively on x axis
dx = (b-a)/(len(u) - 1)   #set the dx discretisation
u[int(0.5/dx):int(1/dx)] = 2    #set the initial condition, t = 0, u = 2 within x = 0.5 and x = 1
x = [i*dx for i in range(0,len(u))]
v = 0.1   #set the viscosity value, notice the CFL stability prerequisite!
t = 0.625   #set the t limit
dt = 0.001  #set the timestep

#du/dx term testing
du_dxx = [((u[j+1]-2*u[j])+u[j-1])/dx**2 for j in range (1,(len(u)-1))]

def Numerical_Diffusion(u,v,t,dt,dx):

    U = []  #tensor to save all the u data spaciotemporal

    for i in range (0,int(t/dt)):       #calculate each timestep
        u_new = u.copy()
        for j in range(1, len(u) - 1):
            u_new[j] = u[j] + v*(dt/dx**2)*((u[j+1]-2*u[j]) + u[j-1])
        u = u_new
        U.append(u_new)

    return U

#function callback
U = Numerical_Diffusion(u,v,t,dt,dx)

#Program to visualization
fig, ax = plt.subplots()
line, = ax.plot(x, U[0], 'r', linewidth=1)
ax.set_xlim(min(x), 2.1)
ax.set_ylim(0, 3)
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
ax.set_title('1D Diffusion')

def animate(n):
    line.set_ydata(U[n])
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(U), interval=30)
plt.show()


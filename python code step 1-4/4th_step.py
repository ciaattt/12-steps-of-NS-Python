import numpy as np
import pandas as pd
import sympy 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from sympy import init_printing
from sympy.utilities.lambdify import lambdify
init_printing(use_latex=True)

element_num = 101 #set the element number in x axis, divide 100 equal length 
a,b = 0,2*np.pi #set the lower and upper bound
dx = (b-a)/(element_num-1)
sigma = 0.1 #we use the stability for the first time regarding to CFL condition sigma should be less than 0.5
to = 0 #set the t init
t_lim = 0.5 #set the t limit
nu_value = 0.07 #set the kinematic viscosity of fluid
dt = sigma * dx**2 / nu_value  #classical stable time step for diffusion
x_vals = np.linspace(a,b,element_num) #

#constuct all the analytical expression based on Prof. Barba's Module 
def Analytical_Burger(x_vals,nu_value,t_lim,to):
    
    x,nu,t = sympy.symbols('x nu t') #set the parameter for constructing the expression
    
    phi = (sympy.exp(-(x - 4 * t)**2 / (4 * nu * (t + 1))) +
           sympy.exp(-(x - 4 * t - 2 * sympy.pi)**2 / (4 * nu * (t + 1)))) #Analytical result of u(x,t) based on burger equation in this case
    dphi_dx = phi.diff(x) #derive the phi with respect to x
    u = -2*nu_value*(dphi_dx/phi) + 4 #determine the u(x,t)

    #insert the x, nu, t value to calculate the  u within the domain
    u_function = lambdify((x,nu,t),u) 

    #this is necessary, so the result shape is (101,) instead of (1,101)
    u_vals_to = np.asarray([u_function(xi,nu_value,to) for xi in x_vals])    #calculate the t0
    u_vals_tlim = np.asarray([u_function(xi,nu_value,t_lim) for xi in x_vals])      #calculate the tlim
  
    return u_vals_to,u_vals_tlim

#u = Analytical_Burger(x,nu,t,to)
u_to = Analytical_Burger(x_vals,nu_value,t_lim,to)[0]
u_tlim = Analytical_Burger(x_vals,nu_value,t_lim,to)[1]

def Numerical_Burger(u_to,nu_value,t_lim,dt,dx):

    U = []  #tensor to save all the u data spaciotemporal

    for i in range (0,int(t_lim/dt)):       #calculate each timestep

        u_new = u_to.copy()

        for j in range(1, len(u_to) - 1):
            
            u_new[j] = (u_to[j] 
                        - u_to[j]*(dt/dx)*(u_to[j]-u_to[j-1]) 
                        + nu_value*(dt/dx**2)*((u_to[j+1]-2*u_to[j]) + u_to[j-1]))


        u_new[0] = (u_to[0] 
                    - u_to[0]*(dt/dx)*(u_to[0]-u_to[-2]) 
                    + nu_value*(dt/dx**2)*((u_to[1]-2*u_to[0]) + u_to[-2]))
        u_new[-1] = u_new[0]
        u_to = u_new
        U.append(u_new)

    return U

#def Numerical_Burger(u_to, nu_value, t_lim, dt, dx):
#    U = []
#    u = u_to.copy()
#    steps = int(t_lim / dt)
#
#    for _ in range(steps):
#        u_new = u.copy()
#        u_new[1:-1] = (
#            u[1:-1] 
#            - u[1:-1] * (dt / dx) * (u[1:-1] - u[:-2]) 
#            + nu_value * (dt / dx**2) * (u[2:] - 2*u[1:-1] + u[:-2])
#        )
#        # Periodic BC
#        u_new[0] = (
#            u[0]
#            - u[0] * (dt / dx) * (u[0] - u[-2])
#            + nu_value * (dt / dx**2) * (u[1] - 2*u[0] + u[-2])
#        )
#        u_new[-1] = u_new[0]  # enforce periodicity
#
#        U.append(u_new.copy())
#        u = u_new
#
#    return U

#function callback
U = Numerical_Burger(u_to,nu_value,t_lim,dt,dx)

#Program to visualization
fig, ax = plt.subplots()
line, = ax.plot(x_vals, U[0], 'r', linewidth=1)
ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
ax.set_title('1D Burger Equation')

def animate(n):
    line.set_ydata(U[n])
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(U), interval=30)
plt.show()
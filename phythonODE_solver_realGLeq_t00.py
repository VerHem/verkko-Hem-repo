# ------------------------------------------------------------------------------------------
# 
# ODE solver for real-valued GL eq based on scipy.integrate.solve_bvp.
# This Python script means to compare and test the 2D and 3D FEM 
# solver.
# ------------------------------------------------------------------------------------------
# author: Quang. Zhang (timohyva@Github). 14. syyskuu. 2022
#


import csv

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------
# RHS of ODE system
# ------------------------------------------------------------------------------------------
def rhs(x, y):

    return np.vstack((y[1], -4.*y[0] + y[0]*y[0]*y[0]))

# ------------------------------------------------------------------------------------------
# BC
# ------------------------------------------------------------------------------------------

def bc(ya, yb):

    return np.array([ya[0], yb[0] - 2.])

a = 0;b = 6  # 6
ini_steps = 14  # 14
plot_steps = 200

x = np.linspace(a, b, ini_steps)
y = np.zeros((2, x.size))

res_a = solve_bvp(rhs, bc, x, y, tol=0.000001, verbose=1)

# ------------------------------------------------------------------------------------------
# import FEM real GL eq result
# ------------------------------------------------------------------------------------------

with open('digtialized_realGLeq_L_shape2.csv', newline='') as f:
     reader = csv.reader(f)
     data = list(reader)

list1 = list(zip(*data))

x_L = []
for ii in list(list1[0]):
    x_L.append(float(ii))

print("x_L is ", x_L)  
    
psi = []
for ii in list(list1[1]):
    psi.append(float(ii))    

print("psi from python is ", psi)  

# ------------------------------------------------------------------------------------------
# plot
# ------------------------------------------------------------------------------------------

x_plot = np.linspace(a, b, plot_steps)
y_plot = res_a.sol(x_plot)[0]
y1_plot = res_a.sol(x_plot)[1]

fig, ax = plt.subplots(1, 1)

ax.plot(x_plot, y_plot, color="blue", label=r'$\psi$')
ax.plot(x_plot, y1_plot, color="orange",label=r'$\partial_{x}\psi$')
ax.scatter(x_L, psi, color="red", marker="o", label=r"$\psi_{FEM}$")

ax.legend(prop={'size': 20});ax.grid(True)
ax.set_xlabel("x", fontsize=20);ax.set_ylabel(r"$\psi\,(\partial_{x}\psi)$", fontsize=20)



plt.show()

# ------------------------------------------------------------------------------------------
# 
# ODE solver for real-valued GL eq based on scipy.integrate.solve_bvp.
# This Python script means to compare and test the 2D and 3D FEM 
# solver.
# ------------------------------------------------------------------------------------------
# author: Quang. Zhang (timohyva@Github). 14. syyskuu. 2022
#


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
# plot
# ------------------------------------------------------------------------------------------

x_plot = np.linspace(a, b, plot_steps)
y_plot = res_a.sol(x_plot)[0]
y1_plot = res_a.sol(x_plot)[1]

fig, ax = plt.subplots(1, 1)

ax.plot(x_plot, y_plot, label=r'$\psi$')
ax.plot(x_plot, y1_plot, label=r'$\partial_{x}\psi$')

ax.legend();ax.grid(True)
ax.set_xlabel("x");ax.set_ylabel(r"$\partial_{x}\psi$")


plt.show()


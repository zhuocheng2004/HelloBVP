import numpy as np
import matplotlib.pyplot as plt
import cheb
import bvp
import solver


def p(x):
    return 0


def q(x):
    return 0


def f(x):
    return 1


n = 4
a, c = 0, 1
bvp_sys = bvp.BVPSystem(a, c, p, q, f, np.array([[1, 0], [0, 1]]))

solver = solver.Solver(bvp_sys, tol=1e-4)

g = lambda x: x*x/2 - x

pts = cheb.points_shifted(100, a, c)

uh, tree, us = solver.solve(pts)
# print(tree)

pts = cheb.points_shifted(100, a, c)
# pts = np.linspace(a, c, num=100)
real_values = list(map(g, pts))

steps = list(range(1, len(us)+1))
interval_nums = np.zeros(len(steps))
errors = np.zeros(len(steps))
for k in range(0, len(us)):
    errors[k] = np.linalg.norm(us[k] - real_values)

plt.plot(pts, real_values)
plt.plot(pts, uh)
plt.show()

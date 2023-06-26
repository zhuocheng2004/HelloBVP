import numpy as np
import matplotlib.pyplot as plt
from .. import bvp, solver


def p(x):
    return 0


def q(x):
    return 0


def f(x):
    return 1


n = 4
a, c = 0, 1
bvp_sys = bvp.BVPSystem(a, c, p, q, f, np.array([[1, 0], [0, 1]]))

solver = solver.Solver(bvp_sys, tol=1e-5)


# The actual solution
def g(x):
    return x*x/2 - x


pts = np.linspace(a, c, num=1000)
real_values = list(map(g, pts))

uh, tree, us = solver.solve(pts)
# print(tree)

steps = list(range(1, len(us)+1))
errors = np.zeros(len(steps))
for k in range(0, len(us)):
    errors[k] = np.linalg.norm(us[k] - real_values)


plt.figure(1)
plt.plot(pts, real_values)
plt.plot(pts, uh)

plt.figure(2)
plt.semilogy(steps, errors)

plt.show()

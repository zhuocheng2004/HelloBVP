import numpy as np
import matplotlib.pyplot as plt
from .. import bvp, solver
from . import helper


def p(x):
    return 0


def q(x):
    return 1


def f(x):
    return 1


nogui = helper.nogui_from_args()

n = 4
a, c = 0, np.pi * 6
bvp_sys = bvp.BVPSystem(a, c, p, q, f, np.array([[1, 0], [0, 1]]))

solver = solver.Solver(bvp_sys, tol=1e-4)


# The actual solution
def g(x):
    return 1 - np.cos(x)


pts = np.linspace(a, c, num=1000)
real_values = list(map(g, pts))

uh, tree, us = solver.solve(pts)
# print(tree)

steps = list(range(1, len(us)+1))
errors = np.zeros(len(steps))
for k in range(0, len(us)):
    errors[k] = np.linalg.norm(us[k] - real_values)


plt.xlabel('x')
plt.ylabel('u(x)')
plt.figure(1)
plt.plot(pts, real_values, label='real solution')
plt.plot(pts, uh, label='computed')
plt.legend(loc='upper right')

plt.savefig('demo_2_1.png')

plt.figure(2)
plt.xlabel('refinement step')
plt.ylabel('error')
plt.semilogy(steps, errors)

plt.savefig('demo_2_2.png')

if not nogui:
    plt.show()

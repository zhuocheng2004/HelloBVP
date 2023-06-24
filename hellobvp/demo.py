import numpy as np
import matplotlib.pyplot as plt
import cheb
import bvp


def zero_f(x):
    return 0


def one_f(x):
    return 1


n = 128
a, c = 0, np.pi * 6
p, q, f = zero_f, one_f, one_f
bvp_sys = bvp.BVPSystem(a, c, p, q, f, np.array([[1, 0], [0, 1]]))
result = bvp_sys.solve_brute(n)
pts = cheb.points_shifted(n, a, c)
plt.plot(pts, result)
plt.plot(pts, list(map(lambda x: 1 - np.cos(x), pts)))
plt.show()


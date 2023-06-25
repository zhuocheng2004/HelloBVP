import numpy as np
import matplotlib.pyplot as plt
import cheb
import bvp
import btree


def zero_f(x):
    return 0


def one_f(x):
    return 1


n = 4
a, c = 0, np.pi * 4
p, q, f = zero_f, one_f, one_f
bvp_sys = bvp.BVPSystem(a, c, p, q, f, np.array([[1, 0], [0, 1]]))

g = lambda x: 1 - np.cos(x)

node = btree.gen_btree_simple(bvp_sys, n, 4)
node.fill()

pts = cheb.points_shifted(100, a, c)
result = node.solve(pts)
plt.plot(pts, result)
plt.plot(pts, list(map(g, pts)))
plt.show()




import numpy as np
import matplotlib.pyplot as plt
import cheb
import bvp
import btree


def zero_f(x):
    return 0


def one_f(x):
    return 1


n = 16
a, c = 0, 1
p, q, f = zero_f, one_f, one_f
bvp_sys = bvp.BVPSystem(a, c, p, q, f, np.array([[1, 0], [0, 1]]))
node = btree.gen_btree_simple(bvp_sys, n, 4)
node.fill()
leaves, pts = btree.get_leaves(node)
xs, vs = [], []
for leaf in leaves:
    xs.extend(cheb.points_shifted(n, leaf.a, leaf.c))
    vs.extend(leaf.sigma_values())
plt.plot(xs, vs)

pts = cheb.points_shifted(n, a, c)
f_values = np.array(list(map(f, pts))).transpose()
mat = bvp_sys.operator_matrix(a, c, n)
sigma = np.linalg.solve(mat, f_values)
plt.plot(pts, sigma)

plt.show()

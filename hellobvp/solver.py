import numpy as np
from . import bvp, btree


class Solver:
    def __init__(self, bvp_system: bvp.BVPSystem, tol: float = 1e-8, n: int = 4, c: int = 4):
        self.bvp_system = bvp_system
        self.n, self.c = n, c
        self.tol = tol
        self.tree = btree.gen_btree_simple(bvp_system, n, 0)
        self.tree.fill()

    def solve_internal(self, xs: np.ndarray) -> np.ndarray:
        return self.tree.solve(xs)

    def refine(self):
        self.tree.foreach_node_up_bottom(lambda node: node.clear_fill_flag())
        leaves = self.tree.get_leaves()[0]
        s_div = btree.get_s_div(self.tree, self.c)
        last_leaf = None
        print("Number of leaves:", len(leaves))
        for leaf in leaves:
            leaf.abd_filled = True
            if leaf.monitor >= s_div:
                # cut a sub-interval into halves
                leaf.abd_filled = False
                a, c = leaf.a, leaf.c
                leaf.left = btree.BTree(self.bvp_system, a, (a + c) / 2, self.n)
                leaf.left.parent = leaf
                leaf.right = btree.BTree(self.bvp_system, (a + c) / 2, c, self.n)
                leaf.right.parent = leaf
            elif last_leaf is not None and last_leaf.parent == leaf.parent \
                    and (last_leaf.monitor + leaf.monitor) < s_div / np.exp2(self.n):
                leaf.parent.remove_childern()  # merge two sub-intervals

            last_leaf = leaf

        # re-compute all parameters
        self.tree.fill()

    def solve(self, xs: np.ndarray) -> tuple:
        us = []
        last_u = np.zeros((1, len(xs)))

        while True:
            u = self.solve_internal(xs)
            us.append(u)
            error_abs = np.linalg.norm(last_u - u, ord=np.inf)
            total = np.linalg.norm(last_u + u, ord=np.inf)
            error_test = error_abs / total
            print('Relative Error:', error_test)
            if error_test > self.tol:
                self.refine()
                last_u = u
            else:
                return u, self.tree, us

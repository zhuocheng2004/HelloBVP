import numpy as np
import cheb
import bvp


class BTree:
    """
    Each instance represents a node in the tree
    """
    def __init__(self, byp_system: bvp.BVPSystem, a: float, c: float, n: int):
        """
        :param byp_system: the BVP system
        :param a: sub interval left
        :param c: sub interval right
        :param n: number of chebyshev points per mesh grid
        """
        self.bvp_system = byp_system
        self.n = n
        self.parent = None
        self.left = self.right = None
        self.a, self.c = a, c
        self.alpha_l = self.alpha_r = 0
        self.beta_l = self.beta_r = 0
        self.delta_l = self.delta_r = 0
        self.mu_l = self.mu_r = 0
        self.abd_filled = self.lambda_filled = False
        self.operator = None
        self.p_inv_phi_l_values, self.p_inv_phi_r_values, self.p_inv_f_values = None, None, None

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return self.left is None and self.right is None

    def is_left(self):
        return not self.is_root() and self == self.parent.left

    def is_right(self):
        return not self.is_root() and self == self.parent.right

    def fill_abd_leaf(self):
        """
        Fill the alpha, beta, delta values for a leaf.
        :return: None
        """
        mat = self.bvp_system.operator_matrix(self.a, self.c, self.n)
        self.operator = mat
        g_l_values = cheb.cheb_values_shifted(lambda x: self.bvp_system.g_l(x), self.a, self.c, self.n)
        g_r_values = cheb.cheb_values_shifted(lambda x: self.bvp_system.g_r(x), self.a, self.c, self.n)
        phi_l_values = cheb.cheb_values_shifted(lambda x: self.bvp_system.phi_l(x), self.a, self.c, self.n)
        self.p_inv_phi_l_values = np.linalg.solve(mat, phi_l_values)
        phi_r_values = cheb.cheb_values_shifted(lambda x: self.bvp_system.phi_r(x), self.a, self.c, self.n)
        self.p_inv_phi_r_values = np.linalg.solve(mat, phi_r_values)
        f_values = cheb.cheb_values_shifted(lambda x: self.bvp_system.f(x), self.a, self.c, self.n)
        self.p_inv_f_values = np.linalg.solve(mat, f_values)
        self.alpha_l = cheb.cheb_quad_raw_shifted(g_l_values * self.p_inv_phi_l_values, self.a, self.c)
        self.alpha_r = cheb.cheb_quad_raw_shifted(g_r_values * self.p_inv_phi_l_values, self.a, self.c)
        self.beta_l = cheb.cheb_quad_raw_shifted(g_l_values * self.p_inv_phi_r_values, self.a, self.c)
        self.beta_r = cheb.cheb_quad_raw_shifted(g_r_values * self.p_inv_phi_r_values, self.a, self.c)
        self.delta_l = cheb.cheb_quad_raw_shifted(g_l_values * self.p_inv_f_values, self.a, self.c)
        self.delta_r = cheb.cheb_quad_raw_shifted(g_r_values * self.p_inv_f_values, self.a, self.c)
        self.abd_filled = True

    def fill_abd(self):
        """
        Recursively fill all alpha, beta, delta values.
        :return: None
        """
        if self.abd_filled:
            return

        if self.is_leaf():
            self.fill_abd_leaf()
        else:
            self.left.fill_abd()
            self.right.fill_abd()
            d = 1 - self.right.alpha_r * self.left.beta_l
            self.alpha_l = \
                (1 - self.right.alpha_l) * (self.left.alpha_l - self.left.beta_l * self.right.alpha_r) / d \
                + self.right.alpha_l
            self.alpha_r = \
                self.right.alpha_r * (1 - self.left.beta_r) * (1 - self.left.alpha_l) / d + self.left.alpha_r
            self.beta_l = \
                self.left.beta_l * (1 - self.right.beta_r) * (1 - self.right.alpha_l) / d + self.right.beta_l
            self.beta_r = \
                (1 - self.left.beta_r) * (self.right.beta_r - self.left.beta_l * self.right.alpha_r) / d \
                + self.left.beta_r
            self.delta_l = (1 - self.right.alpha_l) * self.left.delta_l / d + self.right.delta_l \
                + (self.right.alpha_l - 1) * self.left.beta_l * self.right.delta_r / d
            self.delta_r = (1 - self.left.beta_r) * self.right.delta_r / d + self.left.delta_r \
                + (self.left.beta_r - 1) * self.right.alpha_r * self.left.delta_l / d
            self.abd_filled = True

    def fill_lambda_root(self):
        self.mu_l = self.mu_r = 0
        self.lambda_filled = True

    def fill_lambda(self):
        if self.is_leaf():
            return

        if self.is_root():
            self.fill_lambda_root()

        self.left.mu_l = self.mu_l
        self.right.mu_r = self.mu_r
        v = np.array([
            self.mu_r * (1 - self.right.beta_r) - self.right.delta_r,
            self.mu_l * (1 - self.left.alpha_l) - self.left.delta_l,
        ]).reshape((2, 1))
        m = np.array([
            1, self.right.alpha_r,
            self.left.beta_l, 1
        ]).reshape((2, 2))
        u = np.linalg.solve(m, v)
        self.left.mu_r, self.right.mu_l = u[0, 0], u[1, 0]
        self.left.lambda_filled = self.right.lambda_filled = True

        self.left.fill_lambda()
        self.right.fill_lambda()

    def fill(self):
        self.fill_abd()
        self.fill_lambda()

    def sigma_values(self) -> np.ndarray:
        if not self.is_leaf():
            raise ValueError('only work on leaves')
        if not self.lambda_filled or not self.abd_filled:
            raise ValueError('you should fill parameters first')
        return self.p_inv_f_values + self.mu_l * self.p_inv_phi_l_values + self.mu_r * self.p_inv_phi_r_values

    def solve(self, xs) -> np.ndarray:
        leaves, _ = get_leaves(self)
        result = np.zeros(len(xs))
        for k in range(0, len(xs)):
            x = xs[k]
            s = 0
            for leaf in leaves:
                pts = cheb.points_shifted(self.n, leaf.a, leaf.c)
                green_values = np.array([self.bvp_system.green_func(x, t) for t in pts])
                v = np.multiply(green_values, leaf.sigma_values())
                s += cheb.cheb_quad_raw_shifted(v, leaf.a, leaf.c)
            result[k] = s
        return result

    def param_info(self):
        return f'alpha_l={self.alpha_l}, alpha_r={self.alpha_r} ' \
            + f'beta_l={self.beta_l}, beta_r={self.beta_r} ' \
            + f'delta_l={self.delta_l}, delta_r={self.delta_r} ' \
            + f'mu_l={self.mu_l}, mu_r={self.mu_r} '

    def __str__(self):
        return ("root " if self.is_root() else "") + "{ " + f'left: {self.left}, right: {self.right} ' \
            + self.param_info() \
            + " }"


def gen_btree_simple(bvp_system: bvp.BVPSystem, n: int, depth: int):
    root = BTree(bvp_system, bvp_system.a, bvp_system.c, n)
    level_list = [root]
    for k in range(0, depth):
        parent_level_list = level_list.copy()
        level_list.clear()
        for node in parent_level_list:
            a, c = node.a, node.c
            node.left = BTree(bvp_system, a, (a + c) / 2, n)
            node.left.parent = node
            node.right = BTree(bvp_system, (a + c) / 2, c, n)
            node.right.parent = node
            level_list.append(node.left)
            level_list.append(node.right)
    return root


def get_leaves(node: BTree) -> tuple:
    """
    Get all leaves with the interval points
    with left-to-right traverse
    :param node: root node of tree
    :return: leave nodes and sub interval right-endpoints
    """
    leaves, pts = [], []
    if node.is_leaf():
        return [node], [node.c]
    else:
        result_l = get_leaves(node.left)
        result_r = get_leaves(node.right)
        leaves.extend(result_l[0])
        leaves.extend(result_r[0])
        pts.extend(result_l[1])
        pts.extend(result_r[1])
        return leaves, pts

import numpy as np
import cheb
import bvp


class BTree:
    """
    Each instance represents a node in the tree
    """
    def __init__(self, byp_system: bvp.BVPSystem, n: int):
        """
        :param byp_system: the BVP system
        :param n: number of chebyshev points per mesh grid
        """
        self.bvp_system = byp_system
        self.n = n
        self.parent = None
        self.left = self.right = None
        self.a, self.c = byp_system.a, byp_system.c
        self.alpha_l = self.alpha_r = 0
        self.beta_l = self.beta_r = 0
        self.delta_l = self.delta_r = 0
        self.mu_l = self.mu_r = 0
        self.abd_filled = self.lambda_filled = False

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return self.left is None and self.right is None

    def fill_abd_leaf(self):
        """
        Fill the alpha, beta, delta values for a leaf.
        :return: None
        """
        mat = self.bvp_system.operator_matrix(self.n)
        g_l_values = cheb.cheb_values_shifted(lambda x: self.bvp_system.g_l(x), self.a, self.c, self.n)
        g_r_values = cheb.cheb_values_shifted(lambda x: self.bvp_system.g_r(x), self.a, self.c, self.n)
        phi_l_values = cheb.cheb_values_shifted(lambda x: self.bvp_system.phi_l(x), self.a, self.c, self.n)
        p_inv_phi_l_values = np.linalg.solve(mat, phi_l_values)
        phi_r_values = cheb.cheb_values_shifted(lambda x: self.bvp_system.phi_r(x), self.a, self.c, self.n)
        p_inv_phi_r_values = np.linalg.solve(mat, phi_r_values)
        f_values = cheb.cheb_values_shifted(lambda x: self.bvp_system.f(x), self.a, self.c, self.n)
        p_inv_f_values = np.linalg.solve(mat, f_values)
        self.alpha_l = cheb.cheb_quad_raw_shifted(g_l_values * p_inv_phi_l_values, self.a, self.c)
        self.alpha_r = cheb.cheb_quad_raw_shifted(g_r_values * p_inv_phi_l_values, self.a, self.c)
        self.beta_l = cheb.cheb_quad_raw_shifted(g_l_values * p_inv_phi_r_values, self.a, self.c)
        self.beta_r = cheb.cheb_quad_raw_shifted(g_r_values * p_inv_phi_r_values, self.a, self.c)
        self.delta_l = cheb.cheb_quad_raw_shifted(g_l_values * p_inv_f_values, self.a, self.c)
        self.delta_r = cheb.cheb_quad_raw_shifted(g_r_values * p_inv_f_values, self.a, self.c)
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


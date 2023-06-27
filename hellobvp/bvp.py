import numpy as np
from . import cheb


class BVPSystem:
    """
    Solve 2-point boundary value problem for a second-order Liouville-Sturm System.
    Such a system can be expressed as:
        u''(x) + p(x)u'(x) + q(x)u(x) = f(x)
    on interval [a, c], with homogeneous boundary condition:
        zeta_l0 * u(a) + zeta_l1 * u'(a) = 0
        zeta_r0 * u(c) + zeta_r1 * u'(c) = 0
    """

    def __init__(self, a: float, c: float, p, q, f, zetas):
        """
        :param p:
        :param q:
        :param f:
        :param zetas: a 2x2 matrix consisting of:
            zeta_l0, zeta_l1;
            zeta_r0, zeta_r1
        """
        self.a, self.c = a, c
        self.p, self.q, self.f = p, q, f
        self.zeta_l0, self.zeta_l1 = zetas[0, 0], zetas[0, 1]
        self.zeta_r0, self.zeta_r1 = zetas[1, 0], zetas[1, 1]
        self.s = self.zeta_r0 * self.zeta_l0 * (c - a) + self.zeta_l0 * self.zeta_r1 - self.zeta_r0 * self.zeta_l1

    def g_l(self, x: float) -> float:
        return self.zeta_l0 * (x - self.a) - self.zeta_l1

    def g_r(self, x: float) -> float:
        return self.zeta_r0 * (x - self.c) - self.zeta_r1

    def phi_l(self, x: float) -> float:
        return (self.p(x) * self.zeta_r0 + self.q(x) * self.g_r(x)) / self.s

    def phi_r(self, x: float) -> float:
        return (self.p(x) * self.zeta_l0 + self.q(x) * self.g_l(x)) / self.s

    def green_func(self, x: float, t: float) -> float:
        if x <= t:
            return self.g_l(x) * self.g_r(t) / self.s
        else:
            return self.g_l(t) * self.g_r(x) / self.s

    def operator_matrix(self, b1: float, b2: float, n: int) -> np.ndarray:
        mat = cheb.solve_matrix(lambda x: self.phi_l(x), lambda x: self.phi_r(x),
                                lambda x: self.g_l(x), lambda x: self.g_r(x),
                                b1, b2, n)
        return mat

    def solve_brute(self, n: int) -> np.ndarray:
        """
        No sub-interval refinement,
        solve using the whole interval directly
        :param n: number of chebyshev points
        :return: solution evaluated on cheb points
        """
        pts = cheb.points_shifted(n, self.a, self.c)
        f_values = np.array(list(map(self.f, pts))).transpose()
        mat = self.operator_matrix(self.a, self.c, n)
        sigma = np.linalg.solve(mat, f_values)
        green_mat = np.zeros((n, n))
        result = np.zeros(n)
        for i in range(0, n):
            for j in range(0, n):
                green_mat[i, j] = self.green_func(pts[i], pts[j])
        for i in range(0, n):
            v = np.multiply(green_mat[i, :], sigma)
            result[i] = cheb.cheb_quad_raw_shifted(v, self.a, self.c)
        return result

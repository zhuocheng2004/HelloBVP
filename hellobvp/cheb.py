from functools import partial
import numpy as np


# TODO: doesn't work
cheb_points_cache = {}
cheb_poly_cache = {}


def reset_cache():
    global cheb_points_cache, cheb_poly_cache
    cheb_points_cache = {}
    cheb_poly_cache = {}


def cache_add(cache, n, v):
    if n not in cache:
        cache[n] = v


points_cache_add = partial(cache_add, cheb_points_cache)
poly_cache_add = partial(cache_add, cheb_poly_cache)


def cheb_poly_value(k: int, j: int, n: int):
    """
    generate the value of T_k(t_j)
    :param k: the k-th chebyshev polynomial
    :param j: the j-th cheb point
    :param n: total number of cheb points
    :return: result
    """
    return np.cos(k * (2*n-2*j+1) * np.pi / (2*n))


def points(n: int) -> np.ndarray:
    """
    Generate n Chebyshev points on interval [-1, 1] (does not include endpoints).
    :param n: the number of points we want to generate.
    :return: a numpy row-vector containing n chebyshev points.
    """
    n = int(n)
    # look through the cache
    if n in cheb_points_cache:
        return cheb_points_cache[n]

    if n <= 0:
        raise ValueError(f'Parameter n={n} should not be negative')

    v = np.array([np.cos((2 * n - 2 * k + 1) * np.pi / (2 * n)) for k in range(1, n + 1)])
    points_cache_add(n, v)
    return v


def cheb_expand(f, n: int) -> np.ndarray:
    """
    Interpolate a function on n chebyshev points using linear combination of chebyshev polynomials:
        sum_0^{n-1} a_k T_k(x)
    where
        T_k(cos theta) = cos(k theta)
    are chebyshev polynomials.
    :param f: function to interpolate
    :param n: number of chebyshev points, used in points(n)
    :return: an array consisting of interpolating coefficients a_0 ... a_{n-1}
    """
    ts = points(n)
    result = np.zeros(n)
    result[0] = sum(map(f, ts)) / n
    for k in range(1, n):
        s = 0
        for j in range(1, n+1):
            s += f(ts[j-1]) * cheb_poly_value(k, j, n)
        result[k] = s * 2 / n
    return result


def expand_quad_l(f, n):
    """
    approximate
        int_{-1}^x f(t) dt
    as
        sum_k^{infinity} a_k T_k(x)
    :param f: function
    :param n: number of coefficients to get
    :return: coefficients {a_k} (k = 0, 1, ..., n-1)
    """
    fs = cheb_expand(f, n+1)
    result = np.zeros(n)
    for k in range(2, n):
        result[k] = (fs[k-1] - fs[k+1]) / (2*k)
    if n >= 2:
        result[1] = fs[0] - fs[2] / 2
    s = 0
    for k in range(1, n):
        s += (result[k] if (k % 2 == 1) else -result[k])
    result[0] = s
    return result


def expand_quad_r(f, n):
    """
    approximate
        int_x^1 f(t) dt
    as
        sum_k^{infinity} b_k T_k(x)
    :param f: function
    :param n: number of coefficients to get
    :return: coefficients {b_k} (k = 0, 1, ..., n-1)
    """
    fs = cheb_expand(f, n+1)
    result = np.zeros(n)
    for k in range(2, n):
        result[k] = (fs[k+1] - fs[k-1]) / (2*k)
    if n >= 2:
        result[1] = fs[2] / 2 - fs[0]
    result[0] = -sum(result[1:])
    return result


def matrix_1(n):
    """
    Used for chebyshev expansion
    :param n: number of cheb points
    :return:
    """
    mat = np.zeros((n, n))
    for j in range(0, n):
        mat[0, j] = 1
    for i in range(1, n):
        for j in range(0, n):
            mat[i, j] = 2 * cheb_poly_value(i, j+1, n)
    mat = mat / n
    return mat


def matrix_2l(n):
    """
    f(x) = sum_0^{infinity} f_k T_k(x)
    then
    int_{-1}^x f(t)dt = sum_0^{infinity} a_k T_k(x)
    [a_0 ~ a_{n-1}] = A * [f_0 ~ f_{n-1}]
    A is the returned matrix
    :param n:
    :return: matrix A
    """
    mat = np.zeros((n+1, n+2))
    mat[1, 0], mat[1, 2] = 1, -1/2
    for i in range(2, n+1):
        mat[i, i-1], mat[i, i+1] = 1/(2*i), -1/(2*i)
    for i in range(1, n+1):
        v = mat[i, :]
        mat[0, :] += (v if i % 2 == 1 else -v)

    return mat[:n, :n]


def matrix_2r(n):
    """
    f(x) = sum_0^{infinity} f_k T_k(x)
    then
    int_x^1 f(t)dt = sum_0^{infinity} a_k T_k(x)
    [a_0 ~ a_{n-1}] = A * [f_0 ~ f_{n-1}]
    A is the returned matrix
    :param n:
    :return: matrix A
    """
    mat = np.zeros((n+1, n+2))
    mat[1, 0], mat[1, 2] = -1, 1/2
    for i in range(2, n+1):
        mat[i, i-1], mat[i, i+1] = -1/(2*i), 1/(2*i)
    for i in range(1, n+1):
        mat[0, :] -= mat[i, :]

    return mat[:n, :n]


def matrix_3(n):
    """
    One matrix consisting of chebyshev polynomial values,
    used in the last step of discretization.
    :param n:
    :return: matrix
    """
    mat = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            mat[i, j] = cheb_poly_value(j, i+1, n)
    return mat


def solve_matrix_basic(phi_l, phi_r, g_l, g_r, n):
    """
    discretization of operator:
        (Pu)(x) = u(x) + phi_l(x) int_-1^x g_l(t)u(t)dt + phi_r(x) int_x^1 g_r(t)u(t) dt
    for given values of u at n chebyshev points, say v (an (n+1)-dim column vector),
    values of (Pu) at the chebyshev points can be approximated as Av.
    A is what we return.
    :param phi_l:
    :param phi_r:
    :param g_l:
    :param g_r:
    :param n: number of chebyshev points
    :return: the matrix A
    """
    cheb_pts = points(n)
    d = np.zeros((n, n))
    for i in range(0, n):
        d[i, i] = g_l(cheb_pts[i])
    m = np.matmul(matrix_1(n), d)
    m = np.matmul(matrix_2l(n), m)
    m = np.matmul(matrix_3(n), m)
    d2 = np.zeros((n, n))
    for i in range(0, n):
        d2[i, i] = phi_l(cheb_pts[i])
    m = np.matmul(d2, m)
    m = np.eye(n) + m
    return m

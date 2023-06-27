import numpy as np
import pytest
from hellobvp import cheb
import helper

zero_f = helper.zero_f
one_f = helper.one_f


def test_foo():
    """
    Make sure pytest works.
    """
    assert 1 == 1


def test_points_examples():
    """
    Test by real examples.
    """
    sqrt2 = np.sqrt(2)
    sqrt3 = np.sqrt(3)
    cheb.reset_cache()
    # Call twice to make sure cache works.
    helper.fuzzy_equal_array(cheb.points(1), np.array([0]))
    helper.fuzzy_equal_array(cheb.points(1), np.array([0]))
    helper.fuzzy_equal_array(cheb.points(2), np.array([-sqrt2 / 2, sqrt2 / 2]))
    helper.fuzzy_equal_array(cheb.points(3), np.array([-sqrt3 / 2, 0, sqrt3 / 2]))
    helper.fuzzy_equal_array(cheb.points(3), np.array([-sqrt3 / 2, 0, sqrt3 / 2]))


def test_expand():
    helper.fuzzy_equal_array(cheb.cheb_expand(lambda x: 1, 1), np.array([1]))
    helper.fuzzy_equal_array(cheb.cheb_expand(lambda x: 1, 3), np.array([1, 0, 0]))
    helper.fuzzy_equal_array(cheb.cheb_expand(lambda x: 1, 100), np.array([1] + [0] * 99))
    helper.fuzzy_equal_array(cheb.cheb_expand(lambda x: x, 2), np.array([0, 1]))
    helper.fuzzy_equal_array(cheb.cheb_expand(lambda x: x, 100), np.array([0, 1] + [0] * 98))
    helper.fuzzy_equal_array(cheb.cheb_expand(lambda x: x + 1, 2), np.array([1, 1]))
    helper.fuzzy_equal_array(cheb.cheb_expand(lambda x: 2 * x * x - 1, 3), np.array([0, 0, 1]))
    helper.fuzzy_equal_array(cheb.cheb_expand(lambda x: 2 * x * x, 3), np.array([1, 0, 1]))
    helper.fuzzy_equal_array(cheb.cheb_expand(lambda x: 4 * x * x * x - 3 * x, 4), np.array([0, 0, 0, 1]))
    helper.fuzzy_equal_array(cheb.cheb_expand(lambda x: 4 * x * x * x - 3 * x, 100), np.array([0, 0, 0, 1] + [0] * 96))
    helper.fuzzy_equal_array(cheb.cheb_expand(lambda x: 4 * x * x * x + x + 1, 4), np.array([1, 4, 0, 1]))


def test_expand_quad_l():
    # 0 -> 0
    helper.fuzzy_equal_array(cheb.expand_quad_l(lambda x: 0, 2), np.array([0, 0]))
    helper.fuzzy_equal_array(cheb.expand_quad_l(lambda x: 0, 10), np.array([0] * 10))

    # 1 -> x + 1
    helper.fuzzy_equal_array(cheb.expand_quad_l(lambda x: 1, 2), np.array([1, 1]))
    helper.fuzzy_equal_array(cheb.expand_quad_l(lambda x: 1, 3), np.array([1, 1, 0]))
    helper.fuzzy_equal_array(cheb.expand_quad_l(lambda x: 1, 100), np.array([1, 1] + [0] * 98))

    # 4x -> 2x^2 - 2
    helper.fuzzy_equal_array(cheb.expand_quad_l(lambda x: 4 * x, 3), np.array([-1, 0, 1]))
    helper.fuzzy_equal_array(cheb.expand_quad_l(lambda x: 4 * x, 100), np.array([-1, 0, 1] + [0] * 97))


def test_expand_quad_r():
    # 0 -> 0
    helper.fuzzy_equal_array(cheb.expand_quad_r(lambda x: 0, 2), np.array([0, 0]))
    helper.fuzzy_equal_array(cheb.expand_quad_r(lambda x: 0, 10), np.array([0] * 10))

    # 1 -> 1 - x
    helper.fuzzy_equal_array(cheb.expand_quad_r(lambda x: 1, 2), np.array([1, -1]))
    helper.fuzzy_equal_array(cheb.expand_quad_r(lambda x: 1, 3), np.array([1, -1, 0]))
    helper.fuzzy_equal_array(cheb.expand_quad_r(lambda x: 1, 100), np.array([1, -1] + [0] * 98))

    # 4x -> 2 - 2x^2
    helper.fuzzy_equal_array(cheb.expand_quad_r(lambda x: 4 * x, 3), np.array([1, 0, -1]))
    helper.fuzzy_equal_array(cheb.expand_quad_r(lambda x: 4 * x, 100), np.array([1, 0, -1] + [0] * 97))


def test_solve_basic_01():
    mat = cheb.solve_matrix_basic(zero_f, zero_f, one_f, one_f, 16)
    helper.fuzzy_equal_array(mat, np.eye(16))


def test_solve_basic_02():
    pts = cheb.points(16)
    f, pf = lambda x: x ** 2, lambda x: x ** 2 + 2 / 3
    mat = cheb.solve_matrix_basic(one_f, one_f, one_f, one_f, 16)
    u = np.array(list(map(f, pts))).transpose()
    v = np.array(list(map(pf, pts))).transpose()
    helper.fuzzy_equal_array(v, np.matmul(mat, u))


def test_solve_basic_03():
    pts = cheb.points(16)
    mat = cheb.solve_matrix_basic(one_f, zero_f, one_f, one_f, 16)
    f, pf = lambda x: 1, lambda x: x + 2
    u = np.array(list(map(f, pts))).transpose()
    v = np.array(list(map(pf, pts))).transpose()
    helper.fuzzy_equal_array(v, np.matmul(mat, u))


def test_solve_basic_04():
    pts = cheb.points(16)
    mat = cheb.solve_matrix_basic(one_f, zero_f, one_f, one_f, 16)
    f, pf = lambda x: x, lambda x: x*x/2 + x - 1/2
    u = np.array(list(map(f, pts))).transpose()
    v = np.array(list(map(pf, pts))).transpose()
    helper.fuzzy_equal_array(v, np.matmul(mat, u))


def test_solve_basic_05():
    pts = cheb.points(16)
    mat = cheb.solve_matrix_basic(one_f, zero_f, one_f, one_f, 16)
    f, pf = lambda x: x*x, lambda x: x*x + (x**3 + 1)/3
    u = np.array(list(map(f, pts))).transpose()
    v = np.array(list(map(pf, pts))).transpose()
    helper.fuzzy_equal_array(v, np.matmul(mat, u))


def test_solve_basic_06():
    pts = cheb.points(32)
    mat = cheb.solve_matrix_basic(one_f, zero_f, one_f, one_f, 32)
    f, pf = lambda x: np.cos(x), lambda x: np.cos(x) + np.sin(x) + np.sin(1)
    u = np.array(list(map(f, pts))).transpose()
    v = np.array(list(map(pf, pts))).transpose()
    helper.fuzzy_equal_array(v, np.matmul(mat, u))


def test_solve_01():
    mat = cheb.solve_matrix(zero_f, zero_f, one_f, one_f, 10, 20, 16)
    helper.fuzzy_equal_array(mat, np.eye(16))


def test_solve_02():
    pts = (cheb.points(16) + 1) / 2
    f, pf = lambda x: x ** 2, lambda x: x ** 2 + 1 / 3
    mat = cheb.solve_matrix(one_f, one_f, one_f, one_f, 0, 1, 16)
    u = np.array(list(map(f, pts))).transpose()
    v = np.array(list(map(pf, pts))).transpose()
    helper.fuzzy_equal_array(v, np.matmul(mat, u))


def test_solve_03():
    pts = (cheb.points(16) + 11) / 2
    f, pf = lambda x: x ** 2, lambda x: x ** 2 + 91 / 3
    mat = cheb.solve_matrix(one_f, one_f, one_f, one_f, 5, 6, 16)
    u = np.array(list(map(f, pts))).transpose()
    v = np.array(list(map(pf, pts))).transpose()
    helper.fuzzy_equal_array(v, np.matmul(mat, u), 1e-12)


@pytest.mark.xfail(raises=ValueError)
def test_points_negative():
    """
    Expected failure for negative input.
    """
    cheb.points(-1)

import numpy as np
import pytest
from hellobvp import cheb
import helper


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
    helper.fuzzy_zero(cheb.points(1), np.array([0]))
    helper.fuzzy_zero(cheb.points(1), np.array([0]))
    helper.fuzzy_zero(cheb.points(2), np.array([-sqrt2 / 2, sqrt2 / 2]))
    helper.fuzzy_zero(cheb.points(3), np.array([-sqrt3 / 2, 0, sqrt3 / 2]))
    helper.fuzzy_zero(cheb.points(3), np.array([-sqrt3 / 2, 0, sqrt3 / 2]))


def test_expand():
    helper.fuzzy_zero(cheb.cheb_expand(lambda x: 1, 1), np.array([1]))
    helper.fuzzy_zero(cheb.cheb_expand(lambda x: 1, 3), np.array([1, 0, 0]))
    helper.fuzzy_zero(cheb.cheb_expand(lambda x: 1, 100), np.array([1] + [0] * 99))
    helper.fuzzy_zero(cheb.cheb_expand(lambda x: x, 2), np.array([0, 1]))
    helper.fuzzy_zero(cheb.cheb_expand(lambda x: x, 100), np.array([0, 1] + [0] * 98))
    helper.fuzzy_zero(cheb.cheb_expand(lambda x: x+1, 2), np.array([1, 1]))
    helper.fuzzy_zero(cheb.cheb_expand(lambda x: 2*x*x-1, 3), np.array([0, 0, 1]))
    helper.fuzzy_zero(cheb.cheb_expand(lambda x: 2*x*x, 3), np.array([1, 0, 1]))
    helper.fuzzy_zero(cheb.cheb_expand(lambda x: 4*x*x*x - 3*x, 4), np.array([0, 0, 0, 1]))
    helper.fuzzy_zero(cheb.cheb_expand(lambda x: 4*x*x*x - 3*x, 100), np.array([0, 0, 0, 1] + [0] * 96))
    helper.fuzzy_zero(cheb.cheb_expand(lambda x: 4*x*x*x + x + 1, 4), np.array([1, 4, 0, 1]))


def test_expand_quad_l():
    # 0 -> 0
    helper.fuzzy_zero(cheb.expand_quad_l(lambda x: 0, 2), np.array([0, 0]))
    helper.fuzzy_zero(cheb.expand_quad_l(lambda x: 0, 10), np.array([0] * 10))

    # 1 -> x + 1
    helper.fuzzy_zero(cheb.expand_quad_l(lambda x: 1, 2), np.array([1, 1]))
    helper.fuzzy_zero(cheb.expand_quad_l(lambda x: 1, 3), np.array([1, 1, 0]))
    helper.fuzzy_zero(cheb.expand_quad_l(lambda x: 1, 100), np.array([1, 1] + [0] * 98))

    # 4x -> 2x^2 - 2
    helper.fuzzy_zero(cheb.expand_quad_l(lambda x: 4*x, 3), np.array([-1, 0, 1]))
    helper.fuzzy_zero(cheb.expand_quad_l(lambda x: 4*x, 100), np.array([-1, 0, 1] +[0] * 97))


def test_expand_quad_r():
    # 0 -> 0
    helper.fuzzy_zero(cheb.expand_quad_r(lambda x: 0, 2), np.array([0, 0]))
    helper.fuzzy_zero(cheb.expand_quad_r(lambda x: 0, 10), np.array([0] * 10))

    # 1 -> 1 - x
    helper.fuzzy_zero(cheb.expand_quad_r(lambda x: 1, 2), np.array([1, -1]))
    helper.fuzzy_zero(cheb.expand_quad_r(lambda x: 1, 3), np.array([1, -1, 0]))
    helper.fuzzy_zero(cheb.expand_quad_r(lambda x: 1, 100), np.array([1, -1] + [0] * 98))

    # 4x -> 2 - 2x^2
    helper.fuzzy_zero(cheb.expand_quad_r(lambda x: 4*x, 3), np.array([1, 0, -1]))
    helper.fuzzy_zero(cheb.expand_quad_r(lambda x: 4*x, 100), np.array([1, 0, -1] +[0] * 97))


@pytest.mark.xfail(raises=ValueError)
def test_points_negative():
    """
    Expected failure for negative input.
    """
    cheb.points(-1)

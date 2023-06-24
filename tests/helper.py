import numpy as np


def zero_f(x):
    return 0


def one_f(x):
    return 1


def fuzzy_zero(v1, v2, tol=1e-14):
    assert v1.size == v2.size
    assert np.linalg.norm(v1 - v2) < tol

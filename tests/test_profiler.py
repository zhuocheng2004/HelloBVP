import time

from hellobvp import profiler
import helper


def helper_test_seconds(seconds: float):
    assert seconds >= 0
    profiler.reset()
    name = f'test_second_{seconds}'
    profiler.push(name)
    if seconds > 0:
        time.sleep(seconds)
    profiler.pop(name)
    helper.fuzzy_equal(profiler.get_time(name), seconds, tol=0.1)


def test_zero_seconds():
    helper_test_seconds(0)


def test_zero_seconds_double():
    helper_test_seconds(0)
    helper_test_seconds(0)


def test_one_seconds():
    helper_test_seconds(1)


def test_one_seconds_double():
    helper_test_seconds(1)
    helper_test_seconds(1)


def test_five_seconds():
    helper_test_seconds(5)


def test_two_and_half_seconds():
    helper_test_seconds(2.5)


def test_count():
    profiler.reset()
    profiler.push('test_count')
    for k in range(321):
        profiler.click('test_count')
    profiler.pop('test_count')
    assert profiler.get_count('test_count') == 321 + 1

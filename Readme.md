# A Toy 2-point BVP Problem Solver
For 2023 Numerical Analysis Course Final Project.

See through the `info.tex` for analysis 
and some experiment results.

# Problem
We want to solve 2-point boundary value problem:
```
    u''(x) + p(x)u'(x) + q(x)u(x) = f(x)
```
with homogeneous boundary conditions:
```
    zeta_l0 * u(a) + zeta_l1 * u'(a) = 0
    zeta_r0 * u(c) + zeta_r1 * u'(c) = 0
```

Note that for non-homogeneous problem,
it's easy to convert the system to the sum of two systems,
one is bvp with homogeneous boundary condition, 
the other has a linear or quadratic solution

Our main reference is the article
`A Fast Adaptive Numerical Method for Stiff Two-point Boundary Value Problems`
by `June-Yub Lee` and `Leslie Greengard`. 

This project is a toy program implementing the algorithm suggested by the article.

# System Requirement

Interpreter: `python3`

Dependencies: `numpy`

# Running Demos

```bash
$ python -m hellobvp.demos.demo_sigma
$ python -m hellobvp.demos.demo_1
$ python -m hellobvp.demos.demo_2
$ # These commands can be executed everywhere on your shell
$ # once you installed this package.
```
## Demo Files

### `demo_sigma.py`:
We compare the computed sigma function using two methods:
- whole interval, no refinement
- cut the interval into 2^6=64 parts

### `demo_1.py`, `demo_2.py`:
Demonstration of toy problem 1, 2 
(see description of the two problems at the end)

# Install
```bash
$ python3 -m pip install . # this
$ pip3 install .  # or this
```

# Running Tests
Require pytest installed.

```bash
$ cd tests/
$ pytest
```

# Data for demonstrations and examples
Here is data for two test toy problems which are used 
throughout the development process.
## Demo 1
## Demo 2

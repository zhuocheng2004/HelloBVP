# A Toy 2-point BVP Problem Solver
For 2023 Numerical Analysis Course Final Project.

See through the `info.tex` for analysis 
and some experiment results. 
(you can build it to info.pdf, 
see instructions below)

# Problem
We want to solve 2-point boundary value problem:
```
    u''(x) + p(x)u'(x) + q(x)u(x) = f(x)
```
on interval `[a, c]`

with homogeneous boundary conditions:
```
    zeta_l0 * u(a) + zeta_l1 * u'(a) = 0
    zeta_r0 * u(c) + zeta_r1 * u'(c) = 0
```

Note that for non-homogeneous boundary condition,
it's easy to convert the system to the sum of two systems,
one is bvp with homogeneous boundary condition, 
the other has a linear or quadratic solution

Our main reference is the article
`A Fast Adaptive Numerical Method for Stiff Two-point Boundary Value Problems`
by June-Yub Lee and Leslie Greengard. 

This project is a toy program implementing the algorithm suggested by the article.

# System Requirement

Interpreter: `python3`

Dependencies: `numpy`, `matplotlib`

Recommend running on Linux or similar Unix-like platforms.

However, when implementing in Python, it's not as fast as expected.

# Running Demos

```bash
$ python -m hellobvp.demos.demo_sigma
$ python -m hellobvp.demos.demo_1
$ python -m hellobvp.demos.demo_2
$ # These commands can be executed everywhere on your shell
$ # once you installed this package.
```
or simply
```bash
$ ./run_demos.sh
```

## Demo Files

### `demo_sigma.py`:
We compare the computed sigma function using two methods:
- whole interval, no refinement
- manually cut the interval into 2^6=64 parts using 6-level binary tree

### `demo_1.py`, `demo_2.py`:
Demonstration of toy problem 1, 2 
(see description of the two problems at the end of this file)

# Build PDF

Build the pdf, run make:
```bash
$ make
```
clean, run:
```bash
$ make clean
```

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

Note: `xfail` means `expected failure`.

# Data for demonstrations and examples
Here is data for two test toy problems which are used 
throughout the development process.
## Demo 1
interval: `[0, 1]` 
```
    u''(x) = 1
```
```
    u(0) = u'(1) = 0
```
Solution: `1/2 * (x^2) - x`

## Demo 2
interval: `[0, pi * 6]` 
```
    u''(x) + u(x) = 1
```
```
    u(0) = u'(pi * 6) = 0
```
Solution: `1 - cos(x)`

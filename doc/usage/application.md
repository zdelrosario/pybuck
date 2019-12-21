# Application Guide

---

This is a short description of applications of `pybuck`.

## Setting up an analysis

---

The first stage of performing dimensional analysis is to describe the physical
dimensions of the problem. Using `col_matrix`, we can succinctly define a
dimension matrix. As a running example, we consider the inputs for the *Reynolds
pipe flow problem* [1]. There are five input quantities, described in the table
below.

| Input | Symbol | Units |
|-------|--------|-------|
| Fluid density | $\rho | $\frac{M}{L^3}$ |
| Fluid bulk velocity | $U$ | $\frac{L}{T}$ |
| Pipe diameter | $D$ | $L$ |
| Fluid dynamic viscosity | $\mu$ | $\frac{M}{LT}$ |
| Roughness lengthscale | $\epsilon$ | $L$ |

Expressing this information with `pybuck`, we specify each column of the matrix
as a Python `dict` of non-zero entries.

```python
from pybuck import *

df_dim = col_matrix(
    rho = dict(M=1, L=-3),
    U   = dict(L=1, T=-1),
    D   = dict(L=1),
    mu  = dict(M=1, L=-1, T=-1),
    eps = dict(L=1)
)
df_dim
```

```bash
  rowname  rho  U  D  mu  eps
0       T    0 -1  0  -1    0
1       M    1  0  0   1    0
2       L   -3  1  1  -1    1
```

The *dimension matrix* is now assigned to `df_dim`---each entry is an exponent,
associated with an input and physical dimension. For instance the `rho` column
has the entry `-3` in the `L` row, indicating that `rho` has a factor of
$L^{-3}$ in its physical dimensions.

Note that we did not need to assign the zeros in the matrix, and both variable
and dimension names are provided by keyword argument (not by string). Finally,
note that the ordering of row labels in `rowname` is automatically handled
by `col_matrix()`; for example:

```python
col_matrix(
    U = dict(L=+1, T=-1),
    V = dict(T=-1, L=+1)
)
```

```bash
  rowname   U   V
0       T  -1  -1
1       L   1   1
```

## Buckingham Pi

---

The central result of dimensional analysis is the [buckingham pi
theorem](https://en.wikipedia.org/wiki/Buckingham_%CF%80_theorem). This result
provides a means for a priori dimension reduction; a lossless reduction in the
number of inputs for a physical system. Using the dimension matrix, we can
compute a basis for the set of dimensionless numbers.

```python
df_pi = pi_basis(df_dim)
df_pi
```

```bash
  rowname       pi0       pi1
0     rho -0.521959  0.115207
1       U -0.521959  0.115207
2       D -0.413385 -0.632884
3      mu  0.521959 -0.115207
4     eps -0.108575  0.748091
```

This output indicates that, despite there being five inputs, only two
dimensionless numbers are necessary to fully describe the system.

## Re-expression

---

The dimensionless numbers above are a basis for the pi subspace---the set of all
valid dimensionless numbers for the problem at hand. However, they are fairly
difficult to physically interpret. *Re-expressing* the dimensionless numbers in
a user-selected basis can help us with interpretation.

First, we define a "standard" dimensionless basis.

```python
df_standard = col_matrix(
    Re = dict(rho=1, U=1, D=1, mu=-1), # Reynolds number
    R  = dict(eps=1, D=-1)             # Relative roughness
)
df_standard
```

```bash
  rowname  Re  R
0     rho   1  0
1       U   1  0
2     eps   0  1
3       D   1 -1
4      mu  -1  0
```

`Re` is the Reynolds number, which represents the ratio of inertial to viscous
forces. `R` is the relative roughness, which represents the ratio of roughness
to bulk lengthscales. We can re-express `df_pi` in terms of these standard
numbers to make them more physically interpretable.

```python
df_pi_prime = express(df_pi, df_standard)
df_pi_prime
```

```bash
  rowname       pi0       pi1
0      Re -0.521959  0.115207
1       R -0.108575  0.748091
```

Based on the weights above, we can see that `pi0` is mostly weighted towards
`Re`, while `pi1` is mostly weighted towards `R`. However, both are mixtures
of the two standard dimensionless numbers.

## Empirical Dimension Reduction

---

TODO

## Lurking Variables

---

TODO

## References

---

[1] O. Reynolds, "An experimental investigation of the circumstances which determine whether the motion of water shall be direct or sinuous, and of the law of resistance in parallel channels" (1883) *Royal Society*

[2] Z. del Rosario, M. Lee, and G. Iaccarino, "Lurking Variable Detection via Dimensional Analysis" (2019) *SIAM/ASA Journal on Uncertainty Quantification*

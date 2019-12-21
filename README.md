# Overview

[![Documentation Status](https://readthedocs.org/projects/pybuck/badge/?version=latest)](https://pybuck.readthedocs.io/en/latest/?badge=latest)

This package supports [dimensional
analysis](https://en.wikipedia.org/wiki/Dimensional_analysis) in Python.

## Install

Clone repo and run `python setup.py install`.

## Example usage

Use compact syntax to record the physical dimensions of quantities.

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

Use the dimension matrix `df_dim` to check the physical dimensions of quantities.

```python
df_weights = col_matrix(q = dict(rho=1, U=2))
df_res = inner(df_dim, df_weights)
transpose(df_res)
```

```bash
  rowname  L  M  T
0       q -1  1 -2
```

Use `nondim` to compute the *canonical non-dimensionalizing factor* [Theorem 8.1, 1].

```python
df_flowrate = col_matrix(Q = dict(M=1, L=-3, T=-1))
df_nondim = nondim(df_flowrate, df_dim)
print(inner(df_dim, df_nondim))
df_nondim
```

```bash
  rowname         Q
0     rho  0.571429
1       U  0.571429
2       D -0.714286
3      mu  0.428571
4     eps -0.714286
```

See the
[demo](https://github.com/zdelrosario/pybuck/blob/master/examples/quick_demo.ipynb)
for a quick look at package functionality.

# References

[1] Z. del Rosario, M. Lee, and G. Iaccarino, "Lurking Variable Detection via Dimensional Analysis" (2019) SIAM/ASA Journal on Uncertainty Quantification

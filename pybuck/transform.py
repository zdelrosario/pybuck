__all__ = [
    "express",
    "inner",
    "null",
    "pi_basis"
]

from .core import add_row
from numpy import pad, dot
from numpy.linalg import lstsq, cond
from pandas import DataFrame, Categorical
from scipy.linalg import svd
from scipy import compress
from scipy import transpose as t_mat
import warnings

## Inner product
def inner(df, df_weights, rowname="rowname"):
    """Compute the inner product between two matrices. Matches the
    columns of df to the rows of df_weights.

    :param df: Left matrix
    :param df_weights: Right matrix
    :param rowname: Column name of rownames, default "rowname"

    :type df: DataFrame
    :type df_weights: DataFrame
    :type rowname: string

    :returns: Inner product
    :rtype: DataFrame

    Examples:

    from pybuck import *

    df = col_matrix(
        v = dict(x=+1, y=+1),
        w = dict(x=-1, y=+1)
    )
    df_weights = col_matrix(z = dict(v=1, w=1))

    df_res = inner(df, df_weights)
    df

    """
    ## Check invariants
    if not (rowname in df.columns):
        raise ValueError("df must have {} column".format(rowname))
    if not (rowname in df_weights.columns):
        raise ValueError("df_weights must have {} column".format(rowname))
    if not (set(df_weights[rowname]).issubset(set(df.columns))):
        raise ValueError(
            "df_weights[{}] must be subset of df.columns".format(rowname)
        )

    ## Add rows not in df_weights
    rows_missing = list(
        set(df.drop(rowname, axis=1).columns) - set(df_weights[rowname])
    )
    pack = {
        rows_missing[i]: [0] * (df_weights.shape[1] - 1) \
        for i in range(len(rows_missing))
    }
    df_weights = add_row(df_weights, **pack)

    ## Arrange
    df_weights["_tmp"] = Categorical(
        df_weights[rowname],
        df.drop(rowname, axis=1).columns
    )
    df_weights.sort_values("_tmp", inplace=True)

    A = df.drop(rowname, axis=1).values
    B = df_weights.drop([rowname, "_tmp"], axis=1).values

    ## Compute inner product
    X = dot(A, B)

    ## Gather data
    rownames = df[rowname]
    colnames = df_weights.drop(rowname, axis=1).columns

    data = {rowname: df[rowname]}
    for i in range(X.shape[1]):
        data[colnames[i]] = X[:, i]

    return DataFrame(data)

## Re-expression
def express(df, df_basis, rowname="rowname", ktol=1e6):
    """Express a set of vectors in a target basis. Range (columnspace) is
    considered for re-expression. Equivalent to solving

        Ax = B

    for x with

        A = df_basis.values
        B = df.values

    :param df: Given set of vectors to re-express
    :param df_basis: Given basis
    :param rowname: Column name of rownames, default "rowname"
    :param ktol: Maximum condition number for basis, default 1e6

    :type df: DataFrame
    :type df_basis: DataFrame
    :type rowname: string
    :type ktol: float

    :returns: Set of re-expressed vectors
    :rtype: DataFrame

    Examples:

    from pybuck import *
    df = col_matrix(v = dict(x=1, y=1))
    df_basis = col_matrix(
        v1 = dict(x=+1, y=+1),
        v2 = dict(x=+1, y=-1)
    )

    df_x = express(df_b, df_basis)
    df_x

    """
    ## Check invariants
    if not (rowname in df.columns):
        raise ValueError("df must have {} column".format(rowname))
    if not (rowname in df_basis.columns):
        raise ValueError("df_basis must have {} column".format(rowname))
    if (set(df[rowname]) != set(df_basis[rowname])):
        raise ValueError(
            "{} column must have identical entries in df and df_basis".format(
                rowname
            )
        )

    ## Construct and inspect least squares problem
    A = df_basis.sort_values(rowname) \
                .drop(rowname, axis=1) \
                .reset_index(drop=True) \
                .values
    B = df.sort_values(rowname) \
          .drop(rowname, axis=1) \
          .reset_index(drop=True) \
          .values

    cond_A = cond(A)
    cond_B = cond(B)

    if cond_A > ktol:
        raise ValueError(
            "df_basis ill-conditioned; cond_A = {0:4.3f}".format(
                cond_A
            )
        )
    if cond_B > ktol:
        warnings.warn(
            "df ill-conditioned; cond_B = {0:4.3f}".format(cond_B),
            RuntimeWarning
        )

    ## Solve least squares problem
    res = lstsq(A, B, rcond=None)
    X = res[0]

    ## Organize the output
    col_out = df.drop(rowname, axis=1) \
                .columns
    row_out = df_basis.drop(rowname, axis=1).columns

    data = {rowname: row_out}
    for i in range(len(col_out)):
        data[col_out[i]] = X[:, i]

    return DataFrame(data)

## Nullspace computation
def null(A, eps=1e-15):
    """Computes a basis for the nullspace of a matrix

    :param A: Rectangular matrix
    :param eps: Singular value tolerance for nullspace detection

    :type A: numpy 2d array
    :type eps: float

    :returns: basis for matrix nullspace
    :rtype: numpy 2d array

    Examples:

    from pybuck import null
    import numpy as np

    A = np.arange(9).reshape((3, -1))
    N = null(A)

    """
    u, s, vh = svd(A)
    s = pad(s, (0, vh.shape[0] - len(s)), mode='constant')
    null_mask = (s <= eps)
    null_space = compress(null_mask, vh, axis=0)

    return t_mat(null_space)

## Basis for pi subspace
def pi_basis(df, eps=1e-15, rowname="rowname"):
    """Computes a basis for the pi subspace.

    :param df: Dimension matrix
    :param eps: Singular value tolerance; default = 1e-15
    :param rowname: Column name of rownames, default "rowname"

    :type df: DataFrame
    :type eps: float
    :type rowname: string

    :returns: Basis for pi subspace
    :rtype: DataFrame

    Examples:

    from pybuck import *

    df_dim = col_matrix(
        rho = dict(M=1, L=-3),
        U   = dict(L=1, T=-1),
        D   = dict(L=1),
        mu  = dict(M=1, L=-1, T=-1),
        eps = dict(L=1)
    )

    df_pi = pi_basis(df_dim)
    df_pi

    """
    ## Check invariants
    if not (rowname in df.columns):
        raise ValueError("df must have {} column".format(rowname))

    ## Compute nullspace
    A = df.drop(rowname, axis=1).values
    N = null(A, eps=eps)

    ## Construct dataframe output
    df_return = DataFrame({rowname: df.drop(rowname, axis=1).columns})

    for i in range(N.shape[1]):
        df_return["pi{}".format(i)] = N[:, i]

    return df_return

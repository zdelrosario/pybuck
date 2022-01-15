__all__ = [
    "complete",
    "express",
    "inner",
    "nondim",
    "normalize",
    "null",
    "pi_basis"
]

from .core import add_row, pad_row, transpose
from numpy import pad, dot, number
from numpy.linalg import lstsq, norm, cond, svd
from pandas import DataFrame, Series, Categorical, concat
from scipy import compress
from scipy import transpose as t_mat
import warnings

## Inner product
def inner(df, df_weights, rowname="rowname"):
    """Inner product

    Compute the inner product between two matrices. Matches the columns of df to
    the rows of df_weights.

    Args:
        df (DataFrame): Left matrix
        df_weights (DataFrame): Right matrix
        rowname (str): Rownames which define new column names

    Returns:
        DataFrame: Inner product

    Examples:

        >>> from pybuck import *
        >>> df = col_matrix(
        >>>     v = dict(x=+1, y=+1),
        >>>     w = dict(x=-1, y=+1)
        >>> )
        >>> df_weights = col_matrix(z = dict(v=1, w=1))
        >>> df_res = inner(df, df_weights)
        >>> df

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

    ## Can ignore warning; we're zero-filling entire rows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    """Express vectors in a target basis

    Express a set of vectors in a target basis. Range (columnspace) is
    considered for re-expression. Equivalent to solving

        Ax = B

    for x with

        A = df_basis.values
        B = df.values

    Args:
        df (DataFrame): Given set of vectors to re-express
        df_basis (DataFrame): Given basis
        rowname (str): Rownames which define new column names
        ktol (float): Maximum condition number for basis, default 1e6

    Returns:
        DataFrame: Set of re-expressed vectors

    Examples:

        >>> from pybuck import *
        >>> df = col_matrix(v = dict(x=1, y=1))
        >>> df_basis = col_matrix(
        >>>     v1 = dict(x=+1, y=+1),
        >>>     v2 = dict(x=+1, y=-1)
        >>> )
        >>> df_x = express(df_b, df_basis)
        >>> df_x

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
    """Compute nullspace

    Computes a basis for the nullspace of a matrix.

    Args:
        A (np.array): Rectangular matrix
        eps (float): Singular value tolerance for nullspace detection

    Returns:
        np.array: Basis for matrix nullspace

    Examples:

        >>> from pybuck import null
        >>> import numpy as np
        >>> A = np.arange(9).reshape((3, -1))
        >>> N = null(A)

    """
    u, s, vh = svd(A)
    s = pad(s, (0, vh.shape[0] - len(s)), mode='constant')
    null_mask = (s <= eps)
    null_space = compress(null_mask, vh, axis=0)

    return t_mat(null_space)

## Basis for pi subspace
def pi_basis(df, eps=1e-15, rowname="rowname"):
    """Computes a basis for the pi subspace.

    Take a dimension matrix (as DataFrame) and returns a basis for its nullspace
    (the pi subspace).

    Args:
        df (DataFrame): Dimension matrix
        eps (float): Singular value tolerance; default = 1e-15
        rowname (str): Rownames which define new column names

    Returns:
        DataFrame: Basis for pi subspace

    Examples:

        >>> from pybuck import *
        >>> df_dim = col_matrix(
        >>>     rho = dict(M=1, L=-3),
        >>>     U   = dict(L=1, T=-1),
        >>>     D   = dict(L=1),
        >>>     mu  = dict(M=1, L=-1, T=-1),
        >>>     eps = dict(L=1)
        >>> )
        >>> df_pi = pi_basis(df_dim)
        >>> df_pi

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

## Complete a basis for the pi subspace
def complete(df, df_dim, eps=1e-15, rowname="rowname"):
    """Completes a basis for the pi subspace.

    Take a dimension matrix (as DataFrame) and incomplete basis for the pi
    subspace, and return a basis for its nullspace (the pi subspace). Useful

    Args:
        df (DataFrame): Incomplete basis for pi subspace; columns are vectors of target space
        df_dim (DataFrame): Dimension matrix
        eps (float): Singular value tolerance; default = 1e-15
        rowname (str): Rownames which define new column names

    Returns:
        DataFrame: Basis for pi subspace

    Examples:

        >>> from pybuck import *
        >>> df_dim = col_matrix(
        >>>     rho = dict(M=1, L=-3),
        >>>     U   = dict(L=1, T=-1),
        >>>     D   = dict(L=1),
        >>>     mu  = dict(M=1, L=-1, T=-1),
        >>>     eps = dict(L=1)
        >>> )
        >>> df_pi = pi_basis(df_dim)
        >>> df_pi

    """
    ## Check invariants
    if not (rowname in df.columns):
        raise ValueError("df must have {} column".format(rowname))
    if not (rowname in df_dim.columns):
        raise ValueError("df_dim must have {} column".format(rowname))

    ## Combine data
    df_wk = df_dim.append(transpose(df), sort=True).fillna(0.)

    ## Compute nullspace
    A = df_wk.drop(rowname, axis=1).values
    N = null(A, eps=eps)
    df_new = DataFrame({rowname: df_wk.drop(rowname, axis=1).columns})
    for i in range(N.shape[1]):
        df_new["pi{}".format(i)] = N[:, i]

    ## Construct dataframe output
    df_return = concat(
        (
            df_new.set_index(rowname),
            df.set_index(rowname),
        ),
        axis=1,
        join="outer",
    ).fillna(0.).reset_index()

    return df_return

## Canonical non-dimensionalizing factor
def nondim(df, df_dim, rowname="rowname", ktol=1e6, eps=1e-15):
    """Non-dimensionalize by physical dimensions

    Computes the canonical non-dimensionalizing factor for given physical
    quantities.

    Args:
        df (DataFrame): Dimensions of target quantity, column per quantity
        df_dim (DataFrame): Dimension matrix for physical system
        rowname (str): Rownames which define new column names
        ktol (float): Condition number warning tolerance; default 1e6
        eps (float): Nullspace singular value threshold; default 1e-15

    Returns:
        DataFrame: Canonical non-dimensionalizing factor(s)

    Examples:

        >>> from pybuck import *
        >>> df_dim = row_matrix(q=dict(L=1))
        >>> df = col_matrix(q=dict(L=1))
        >>> nondim(df, df_dim)

    References:
        Z. del Rosario, M. Lee, and G. Iaccarino, "Lurking Variable Detection via Dimensional Analysis" (2019) SIAM/ASA Journal on Uncertainty Quantification (Theorem 8.1)

    """
    ## Check invariants
    if not (rowname in df.columns):
        raise ValueError("df must have {} column".format(rowname))
    if not (rowname in df_dim.columns):
        raise ValueError("df_dim must have {} column".format(rowname))
    if not (set(df[rowname]).issubset(set(df_dim[rowname]))):
        raise ValueError("df[rowname] must be subset of df_dim[rowname]")

    ## Set up linear system
    df_null = pi_basis(df_dim, eps=eps, rowname=rowname)
    df_stacked = concat(
        (df_dim, transpose(df_null)),
        axis=0,
        sort=False,
        ignore_index=True
    )

    ## Negate and pad
    cols = df.drop(rowname, axis=1).columns
    df[cols] = -df[cols]
    df = pad_row(df, df_stacked)

    ## Solve
    df_res = express(df, df_stacked, rowname="rowname", ktol=1e6)

    return df_res

## Normalize columns
def normalize(df):
    """Length-normalize numeric columns

    Length-normalizes the numeric columns of a given dataframe.

    Args:
        df (DataFrame): Dimensions of target quantity, column per quantity

    Returns:
        DataFrame: Data with normalized columns

    Examples:

        >>> from pybuck import *
        >>> df = col_matrix(
        >>>     v1=dict(x=+1, y=+1),
        >>>     v2=dict(x=+1, y=-1)
        >>> )
        >>> normalize(df))

    """
    cols_numeric = df.select_dtypes(include=number).columns.tolist()
    lengths = Series(
        data=norm(df[cols_numeric].values, axis=0),
        index=cols_numeric
    )

    df_res = df.copy()
    for col in cols_numeric:
        df_res[col] = df_res[col] / lengths[col]

    return df_res

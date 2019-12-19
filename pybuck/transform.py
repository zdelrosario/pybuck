__all__ = [
    "express"
]

from numpy.linalg import lstsq, cond
from pandas import DataFrame
import warnings

## Re-expression
def express(df, df_basis, rowname="rowname", ktol=1e6):
    """Express a set of vectors in a target basis. Range (columnspace) is
    considered for re-expression. Equivalent to solving
        Ax = B

    for x with
        A = df_basis.values
        B = df.values

    @param df Given set of vectors to re-express
    @param df_basis Given basis
    @param rowname Column name of rownames, default "rowname"
    @param ktol Maximum condition number for basis, default 1e6

    @type df DataFrame
    @type df_basis DataFrame
    @type rowname string
    @type ktol float

    Examples:

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

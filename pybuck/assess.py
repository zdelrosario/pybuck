__all__ = [
    "angles"
]

from scipy.linalg import subspace_angles

## Assess subspace angles
def angles(df1, df2, rowname="rowname"):
    """Compute the subspace angles between two matrices.
    A wrapper for scipy.linalg.subspace_angles that corrects
    for row and column ordering.

    :param df1: First matrix to compare
    :param df2: Second matrix to compare
    :param rowname: Column name of rownames, default "rowname"
                   Must be same for df1 and df2

    :type df1: DataFrame
    :type df2: DataFrame
    :type rowname: string

    :returns: array of angles (in radians)
    :rtype: numpy array

    Examples:

    from pybuck import *

    df = col_matrix(v = dict(x=+1, y=+1))

    df_v1 = col_matrix(w = dict(x=+1, y=-1))
    df_v2 = col_matrix(w = dict(x=+1, y=+1))

    theta1 = angles(df, df_v1)
    theta2 = angles(df, df_v2)

    """
    ## Check invariants
    if not (rowname in df1.columns):
        raise ValueError("df1 must have {} column".format(rowname))
    if not (rowname in df2.columns):
        raise ValueError("df2 must have {} column".format(rowname))

    ## Compute subspace angles
    A1 = df1.sort_values(rowname) \
            .drop(rowname, axis=1) \
            .values
    A2 = df2.sort_values(rowname) \
            .drop(rowname, axis=1) \
            .values

    return subspace_angles(A1, A2)

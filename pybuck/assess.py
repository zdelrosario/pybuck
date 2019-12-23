__all__ = [
    "angles"
]

from scipy.linalg import subspace_angles

## Assess subspace angles
def angles(df1, df2, rowname="rowname"):
    """Subspace angles

    Compute the subspace angles between two matrices. A wrapper for
    scipy.linalg.subspace_angles that corrects for row and column ordering.

    Args:
        df1 (DataFrame): First matrix to compare
        df2 (DataFrame): Second matrix to compare
        rowname (str): Rownames which define new column names
            Must be same for df1 and df2

    Returns:
        np.array: Array of angles (in radians)

    Examples:

    from pybuck import *
        >>> df = col_matrix(v = dict(x=+1, y=+1))
        >>> df_v1 = col_matrix(w = dict(x=+1, y=-1))
        >>> df_v2 = col_matrix(w = dict(x=+1, y=+1))
        >>> theta1 = angles(df, df_v1)
        >>> theta2 = angles(df, df_v2)

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

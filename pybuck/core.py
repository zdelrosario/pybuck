__all__ = [
    "col_matrix",
    "row_matrix",
    "gather",
    "spread",
    "transpose"
]

from numpy import nan
from pandas import DataFrame, melt

## Helper functions
# --------------------------------------------------
def gather(df, key, value, cols):
    id_vars = [col for col in df.columns if col not in cols]
    id_values = cols
    var_name = key
    value_name = value

    return melt(df, id_vars, id_values, var_name=var_name, value_name=value_name)

def spread(df, key, value, fill=nan, drop=False):
    index = [col for col in df.columns if ((col != key) and (col != value))]

    df_new = df.pivot_table(
        index=index,
        columns=key,
        values=value,
        fill_value=fill
    ).reset_index()

    ## Drop extraneous info
    df_new = df_new.rename_axis(None, axis=1)
    if drop:
        df_new.drop("index", axis=1, inplace=True)

    return df_new

def transpose(df, rowname="rowname"):
    """
    Transpose a dataframe around a single `rowname` column.

    @param df Matrix to transpose, must have column `rowname`
    @param rowname Rownames which define new column names

    @type df DataFrame
    @type rowname string

    @returns Transposed result
    @rtype DataFrame

    Examples:

    from pybuck import *

    df = col_matrix(x=pencil(a=1, b=1), y=pencil(a=-1, y=-1))
    df
    # >>>   rowname  x  y
    # >>> 0       a  1 -1
    # >>> 1       b  1 -1

    transpose(df)
    # >>>   rowname  a  b
    # >>> 0       x  1  1
    # >>> 1       y -1 -1

    """
    cols = [col for col in df.columns if col != rowname]

    df_gathered = gather(
        df,
        key="key",
        value="value",
        cols=cols
    )
    df_t = spread(df_gathered, rowname, "value")
    df_t.rename({"key": rowname}, axis=1, inplace=True)

    return df_t

## Constructor functions
# --------------------------------------------------
def col_matrix(**kwargs):
    """Create a matrix via column construction. Automatically fills zero entries.
    Intended for use with dict().

    @param col Name of col
    @type col dict

    @returns Dense matrix
    @rtype DataFrame

    Examples:

    from pybuck import *
    df_dim = col_matrix(
        rho = dict(M=1, L=-3),
        U   = dict(L=1, T=-1),
        D   = dict(L=1),
        mu  = dict(M=1, L=-1, T=-1),
        eps = dict(L=1)
    )
    """
    ## Get full list of rows and columns
    cols = []
    rows = set()

    for col, row in kwargs.items():
        rows = rows.union(set(row.keys()))
        cols.append(col)
    rows = list(rows)
    row_ind = dict([(rows[i], i) for i in range(len(rows))])

    ## Build matrix
    n_rows = len(rows)
    data = {"rowname": rows}

    for col in cols:
        col_values = [0] * n_rows
        for row in kwargs[col].keys():
            col_values[row_ind[row]] = kwargs[col][row]
        data[col] = col_values

    return DataFrame(data)

def row_matrix(**kwargs):
    """Create a matrix via row construction. Automatically fills zero entries.
    Intended for use with dict().

    @param row Name of row
    @type row dict

    @returns Dense matrix
    @rtype DataFrame

    Examples:

    from pybuck import *
    df_pi = row_matrix(
        Re = dict(rho=1, U=1, D=1, mu=-1),
        R  = dict(D=1, eps=-1)
    )
    """
    df_col = col_matrix(**kwargs)
    return transpose(df_col)

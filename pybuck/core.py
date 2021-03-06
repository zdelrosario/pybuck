__all__ = [
    "add_col",
    "add_row",
    "col_matrix",
    "row_matrix",
    "gather",
    "pad_row",
    "spread",
    "transpose"
]

from numpy import nan
from pandas import DataFrame, melt, merge
import warnings

## Helper functions
# --------------------------------------------------
def gather(df, key, value, cols):
    """Makes a DataFrame longer by gathering columns.

    """
    id_vars = [col for col in df.columns if col not in cols]
    id_values = cols
    var_name = key
    value_name = value

    return melt(df, id_vars, id_values, var_name=var_name, value_name=value_name)

def spread(df, key, value, fill=nan, drop=False):
    """Makes a DataFrame wider by spreading columns.

    """
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
    """Transpose a DataFrame

    Transpose a DataFrame around a single `rowname` column.

    Args:
        df (DataFrame): Matrix to transpose, must have column `rowname`
        rowname (str): Rownames which define new column names

    Returns:
        DataFrame: Transposed result

    Examples:

        >>> from pybuck import *
        >>> df = col_matrix(x=dict(a=1, b=1), y=dict(a=-1, y=-1))
        >>> df
          rowname  x  y
        0       a  1 -1
        1       b  1 -1
        >>> transpose(df)
          rowname  a  b
        0       x  1  1
        1       y -1 -1

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
def col_matrix(rowname="rowname", **kwargs):
    """Matrix via column construction

    Create a matrix via column construction. Automatically fills zero entries.
    Intended for use with dict().

    Args:
        rowname (str): Name of rowname column; default = "rowname"
        col (dict): Column definition. Column name inferred from keyword.
            Column entries given by dictionary.

    Returns:
        DataFrame: Dense matrix

    Examples:

        >>> from pybuck import *
        >>> df_dim = col_matrix(
        >>>     rho = dict(M=1, L=-3),
        >>>     U   = dict(L=1, T=-1),
        >>>     D   = dict(L=1),
        >>>     mu  = dict(M=1, L=-1, T=-1),
        >>>     eps = dict(L=1)
        >>> )

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
    data = {rowname: rows}

    for col in cols:
        col_values = [0] * n_rows
        for row in kwargs[col].keys():
            col_values[row_ind[row]] = kwargs[col][row]
        data[col] = col_values

    return DataFrame(data)

def row_matrix(rowname="rowname", **kwargs):
    """Matrix via row construction

    Create a matrix via row construction. Automatically fills zero entries.
    Intended for use with dict().

    Args:
        rowname (str): Name of rowname column; default = "rowname"
        row (dict): Row definition. Row name inferred from keyword.
            Row entries given by dictionary.

    Returns:
        DataFrame: Dense matrix

    Examples:

        >>> from pybuck import *
        >>> df_pi = row_matrix(
        >>>     Re = dict(rho=1, U=1, D=1, mu=-1),
        >>>     R  = dict(D=1, eps=-1)
        >>> )

    """
    df_col = col_matrix(rowname=rowname, **kwargs)
    return transpose(df_col)

def add_col(df, rowname="rowname", **kwargs):
    """Add a column

    Add a column to a DataFrame, matching existing rownames.

    Args:
        df (DataFrame): Data to mutate
        rowname (str): Rownames to match; default = "rowname"
        col (dict): Column to add; name inferred from keyword. May provide
            as array of proper length or as dict.

    Returns:
        DataFrame: Original df with added columns

    :pre: (len(col) == df.shape[0]) | isinstance(col, dict)

    Examples:

        >>> from pybuck import *
        >>> df = col_matrix(rho = dict(M=1, L=-3), U = dict(L=1, T=-1))
        >>> df = add_col(df, D=dict(L=1), v=[0,0,1])

    """
    ## Check invariants
    if not (rowname in df.columns):
        raise ValueError("df must have {} column".format(rowname))

    df_return = df.copy()

    for key, value in kwargs.items():
        if isinstance(value, dict):
            df_tmp = DataFrame({
                rowname: list(value.keys()),
                key: list(value.values())
            })

            df_return = merge(
                df_return,
                df_tmp,
                on=rowname,
                how="left"
            ).fillna(value=0)
        else:
            warnings.warn(
                "Assuming {0:} ordering for {1:}...".format(rowname, key),
                RuntimeWarning
            )
            df_return[key] = value

    return df_return

def add_row(df, rowname="rowname", **kwargs):
    """Add a column to a DataFrame, matching existing rownames.

    Args:
        df (DataFrame): Data to mutate
        rowname (str): Rownames to match; default = "rowname"
        row (dict): Row to add; name inferred from keyword. May provide
                    as array of proper length or as dict.

    Returns:
        DataFrame: Original df with added rows

    :pre: (len(row) == df.shape[1]) | isinstance(col, dict)

    Examples:

        >>> from pybuck import *
        >>> df = row_matrix(v = dict(x=1, y=1, z=1))
        >>> df = add_row(df, w = dict(y=-1))
        >>> df

    """
    ## Implement in terms of add_col()
    df_tmp = transpose(df, rowname=rowname)
    df_tmp = add_col(df_tmp, rowname=rowname, **kwargs)

    return transpose(df_tmp, rowname=rowname)

def pad_row(df, df_ref, rowname="rowname"):
    """Pad DataFrame to match a reference

    Pad a target DataFrame with zero-rows to match a reference.

    Args:
        df (DataFrame): Data to pad
        df_ref (DataFrame): Reference for padding

    Returns:
        DataFrame: Row-padded dataframe

    Examples:
        >>> from pybuck import *
        >>> df = row_matrix(u=dict(x=1, y=1))
        >>> df_ref = row_matrix(
        >>>      u=dict(x=1, y=1),
        >>>      v=dict(x=0, y=0)
        >>> )
        >>> pad_row(df, df_ref)

    """
    ## Check invariants
    if not (rowname in df.columns):
        raise ValueError("df must have {} column".format(rowname))
    if not (rowname in df_ref.columns):
        raise ValueError("df_ref must have {} column".format(rowname))

    ## Add rows not in df
    rows_missing = list(set(df_ref[rowname]) - set(df[rowname]))
    pack = {
        rows_missing[i]: [0] * (df.shape[1] - 1) \
        for i in range(len(rows_missing))
    }

    ## Can ignore warning; we're zero-filling entire rows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = add_row(df, **pack)

    return df

# --------------------------------------------------
def stringexp(df, rowname="rowname", mode="plain"):
    """Exponentiate and stringify a DataFrame of exponents.

    :param df: Data to exponentiate and stringify
    :param rowname: Name of rowname column; default = "rowname"
    :param mode:
    """

import unittest
import pandas as pd

from context import pybuck as bu

# --------------------------------------------------
class TestConstructors(unittest.TestCase):
    def setUp(self):
        self.df_dim = pd.DataFrame({
            "rowname": ["L", "M", "T"],
            "rho":     [ -3,   1,   0],
            "U":       [  1,   0,  -1],
            "D":       [  1,   0,   0],
            "mu":      [ -1,   1,  -1],
            "eps":     [  1,   0,   0]
        }).sort_values("rowname").reset_index(drop=True)

        self.df_pi = pd.DataFrame({
            "rowname": ["Re", "R"],
            "rho":     [   1,   0],
            "U":       [   1,   0],
            "D":       [   1,  -1],
            "mu":      [  -1,   0],
            "eps":     [   0,   1]
        }).sort_values("rowname").reset_index(drop=True)

    def test_add_col(self):
        df_ful = bu.col_matrix(
            rho = dict(M=1, L=-3),
            U   = dict(L=1, T=-1),
            D   = dict(L=1)
        ).sort_values("rowname") \
         .reset_index(drop=True)

        df_sub = bu.col_matrix(rho = dict(M=1, L=-3), U = dict(L=1, T=-1))

        df_res = bu.add_col(df_sub, D=dict(L=1)) \
                   .sort_values("rowname") \
                   .reset_index(drop=True)

        pd.testing.assert_frame_equal(
            df_ful,
            df_res,
            check_exact=False,
            check_dtype=False,
            check_column_type=False
        )

        with self.assertRaises(ValueError):
            bu.add_col(pd.DataFrame(), D=dict(M=1))

        with self.assertWarns(Warning):
            bu.add_col(df_sub, D=[0, 1, 0])

    def test_add_row(self):
        df = bu.row_matrix(v = dict(x=1, y=1))
        df_res = bu.add_row(df, w = dict(y=1))

        df_true = bu.row_matrix(
            v = dict(x=1, y=1),
            w = dict(y=1)
        )

        pd.testing.assert_frame_equal(
            df_res,
            df_true,
            check_exact=False,
            check_dtype=False,
            check_column_type=False
        )

    def test_col_matrix(self):
        df_dim = bu.col_matrix(
            rho = dict(M=1, L=-3),
            U   = dict(L=1, T=-1),
            D   = dict(L=1),
            mu  = dict(M=1, L=-1, T=-1),
            eps = dict(L=1)
        ).sort_values("rowname").reset_index(drop=True)

        self.assertTrue(
            self.df_dim[df_dim.columns].equals(df_dim)
        )

    def test_row_matrix(self):
        df_pi = bu.row_matrix(
            Re = dict(rho=1, U=1, D=1, mu=-1),
            R  = dict(eps=1, D=-1)
        ).sort_values("rowname").reset_index(drop=True)

        self.assertTrue(
            self.df_pi[df_pi.columns].equals(df_pi)
        )

    def test_rowname(self):
        df_tmp = bu.col_matrix(x=dict(a=1), rowname="foo")

        df_ref = pd.DataFrame({
            "foo": ["a"],
            "x":   [  1]
        })

        self.assertTrue(df_ref.equals(df_tmp))

# --------------------------------------------------
class TestReshape(unittest.TestCase):
    def setUp(self):
        ## Transpose test
        self.df_rownames = pd.DataFrame({
            "rowname": ["a", "b"],
            "x":       [0, 1],
            "y":       [2, 3],
        })
        self.df_transposed = pd.DataFrame({
            "rowname": ["x", "y"],
            "a":       [ 0, 2],
            "b":       [ 1, 3]
        })

        ## Gather test
        self.df_wide = pd.DataFrame({
            "a": [0],
            "b": [1]
        })
        self.df_gathered = pd.DataFrame({
            "key": ["a", "b"],
            "value": [0, 1]
        })

        ## Spread test
        self.df_long = pd.DataFrame({
            "key":   ["a", "b"],
            "value": [  0,   1]
        })
        self.df_spreaded = pd.DataFrame({
            "index": ["value"], "a": [0], "b": [1]
        })
        self.df_spreaded_drop = pd.DataFrame({
            "a": [0], "b": [1]
        })

    def test_transpose(self):
        self.assertTrue(
            self.df_transposed.equals(
                bu.transpose(self.df_rownames)
            )
        )

    def test_gather(self):
        self.assertTrue(
            self.df_gathered.equals(
                bu.gather(self.df_wide, "key", "value", ["a", "b"])
            )
        )

    def test_spread(self):
        self.assertTrue(
            self.df_spreaded.equals(
                bu.spread(self.df_long, "key", "value")
            )
        )

        self.assertTrue(
            self.df_spreaded_drop.equals(
                bu.spread(self.df_long, "key", "value", drop=True)
            )
        )

## Run tests
if __name__ == "__main__":
    unittest.main()

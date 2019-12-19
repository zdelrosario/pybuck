import unittest
import pandas as pd

from context import pybuck as bu

# --------------------------------------------------
class TestExpress(unittest.TestCase):
    def setUp(self):
        self.df_basis = \
            bu.col_matrix(
                v1 = dict(x=1, y=1),
                v2 = dict(x=1, y=-1)
            )

        self.df_basis_ill = \
            bu.col_matrix(
                v1 = dict(x=1, y=1),
                v2 = dict(x=1, y=1 - 1e-9)
            )

        self.df_b1 = bu.col_matrix(
            v = dict(x=1., y=1.)
        ).sort_values("rowname").reset_index(drop=True)

        self.df_b1_ill = bu.col_matrix(
            v = dict(x=1., y=1.),
            w = dict(x=1., y=1.)
        ).sort_values("rowname").reset_index(drop=True)

        self.df_x1 = bu.col_matrix(
            v = dict(v1=1., v2=0.)
        ).sort_values("rowname").reset_index(drop=True)

    def test_express_correctness(self):
        df_res = bu.express(
            self.df_b1,
            self.df_basis
        ).sort_values("rowname").reset_index(drop=True)

        pd.testing.assert_frame_equal(
            df_res,
            self.df_x1,
            check_exact=False
        )

    def test_express_raises(self):
        with self.assertRaises(ValueError):
            bu.express(pd.DataFrame(), self.df_basis)

        with self.assertRaises(ValueError):
            bu.express(self.df_b1, pd.DataFrame())

        with self.assertRaises(ValueError):
            bu.express(
                pd.DataFrame({"rowname": ["a", "b", "c"]}),
                self.df_basis
            )

    def test_express_cond(self):
        with self.assertRaises(ValueError):
            bu.express(self.df_b1, self.df_basis_ill)

        with self.assertWarns(Warning):
            bu.express(self.df_b1_ill, self.df_basis)

## Run tests
if __name__ == "__main__":
    unittest.main()

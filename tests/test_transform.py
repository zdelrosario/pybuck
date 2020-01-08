import unittest
import numpy as np
import pandas as pd

from context import pybuck as bu

# --------------------------------------------------
class TestInner(unittest.TestCase):
    def setUp(self):
        pass

    def test_inner(self):
        df = bu.col_matrix(
            v = dict(x=+1, y=+1),
            w = dict(x=-1, y=+1)
        )

        # Full weights
        df_w1 = bu.col_matrix(z = dict(v=1, w=1))
        df_r1 = bu.inner(df, df_w1)
        df_t1 = bu.col_matrix(z = dict(x=0, y=2))

        # Missing weights
        df_w2 = bu.col_matrix(z = dict(v=1))
        df_r2 = bu.inner(df, df_w2)
        df_t2 = bu.col_matrix(z = dict(x=+1, y=+1))

        pd.testing.assert_frame_equal(
            df_r1,
            df_t1,
            check_exact=False
        )

        pd.testing.assert_frame_equal(
            df_r2,
            df_t2,
            check_exact=False
        )

        with self.assertRaises(ValueError):
            bu.inner(pd.DataFrame(), df_w1)

        with self.assertRaises(ValueError):
            bu.inner(df, pd.DataFrame())

        with self.assertRaises(ValueError):
            bu.inner(bu.col_matrix(w=dict(x=1, y=1)), df_w1)

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

# --------------------------------------------------
class TestNull(unittest.TestCase):
    def setUp(self):
        self.A = np.array([
            [ 1, 0, 0],
            [ 0, 1, 0]
        ])
        self.N = np.array([
            [ 0],
            [ 0],
            [ 1]
        ])

    def test_null(self):
        res = bu.null(self.A)
        self.assertTrue(np.array_equal(res, self.N))

# --------------------------------------------------
class TestPiBasis(unittest.TestCase):
    def setUp(self):
        self.df_dim = bu.col_matrix(
            x = dict(M=1),
            y = dict(M=1)
        )
        self.df_pi = bu.col_matrix(
            pi0 = dict(x=-1 / np.sqrt(2), y=+1 / np.sqrt(2))
        )

    def test_pi_basis(self):
        df_res = bu.pi_basis(self.df_dim)

        self.assertTrue(
            np.isclose(bu.angles(df_res, self.df_pi), 0)
        )

        with self.assertRaises(ValueError):
            bu.pi_basis(pd.DataFrame())

# --------------------------------------------------
class TestNondim(unittest.TestCase):
    def setUp(self):
        pass

    def test_nondim(self):
        df_dim = bu.row_matrix(L = dict(u=1, v=1))
        df = bu.col_matrix(q = dict(L=1))
        df_res = bu.nondim(df, df_dim) \
                   .sort_values("rowname") \
                   .reset_index(drop=True)

        df_true = bu.col_matrix(q = dict(u=-0.5, v=-0.5)) \
                    .sort_values("rowname") \
                    .reset_index(drop=True)

        pd.testing.assert_frame_equal(
            df_res,
            df_true,
            check_exact=False
        )

        with self.assertRaises(ValueError):
            bu.nondim(pd.DataFrame(), df_dim)
        with self.assertRaises(ValueError):
            bu.nondim(df, pd.DataFrame())
        with self.assertRaises(ValueError):
            bu.nondim(bu.col_matrix(q = dict(T=1)), df_dim)

# --------------------------------------------------
class TestNormalize(unittest.TestCase):
    def setUp(self):
        pass

    def test_normalize(self):
        df_target = bu.col_matrix(v=dict(x=1, y=1))
        df_true = bu.col_matrix(v=dict(x=1/np.sqrt(2), y=1/np.sqrt(2)))

        df_res = bu.normalize(df_target)

        self.assertTrue(df_true.equals(df_res))

## Run tests
if __name__ == "__main__":
    unittest.main()

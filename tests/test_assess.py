import unittest
import numpy as np
import pandas as pd

from context import pybuck as bu

# --------------------------------------------------
class TestAssess(unittest.TestCase):
    def setUp(self):
        self.df = bu.col_matrix(v = dict(x=+1, y=+1))

        self.df_v1 = bu.col_matrix(w = dict(x=+1, y=-1))
        self.df_v2 = bu.col_matrix(w = dict(x=+1, y=+1))

    def test_angles(self):
        theta1 = bu.angles(self.df, self.df_v1)
        theta2 = bu.angles(self.df, self.df_v2)

        self.assertTrue(np.isclose(theta1, np.pi/2))
        self.assertTrue(np.isclose(theta2, 0))

        with self.assertRaises(ValueError):
            bu.angles(pd.DataFrame(), self.df_v1)

        with self.assertRaises(ValueError):
            bu.angles(self.df, pd.DataFrame())

## Run tests
if __name__ == "__main__":
    unittest.main()

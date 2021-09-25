### Empirical Drag Law, from Tavanashad et al. (2021)
__all__ = [
    "dim_mat",
    "fcn_drag",
    "Q_names",
    "Q_lo",
    "Q_hi",
]

from numpy import array

#     0,     1,     2,   3,    4,   5
# rho_f, rho_p, W_avg, d_p, mu_f, phi
dim_mat = array([
    [ 1,  1,  0,  0,  1,  0], # M
    [-3, -3,  1,  1, -1,  0], # L
    [ 0,  0, -1,  0, -1,  0], # T
])

Q_names = ["rho_f", "rho_p", "W_avg", "d_p", "mu_f", "phi"]
# Parameter domain
#             rho_f, rho_p, W_avg,  d_p, mu_f, phi
Q_lo = array([0.9e3,   1e1,   1e0, 1e-3, 1e-4, 0.1])
Q_hi = array([1.1e3,   1e5,   1e1, 1e-1, 2e-3, 0.4])

# model coefficients
c1 = 0.245
c2 = 22.8
c3 = 0.242
c4 = 130.371
c5 = 6.708
c6 = 0.233
c7 = 140.272
c8 = 2.299

# Drag law (dimensionless, by construction)
def fcn_drag(q):
    r"""Evaluate empirical drag law
    
    Args:
        q = [rho_f, rho_p, W_avg, d_p, mu_f, phi]
        where
        rho_f = fluid density
        rho_p = particle density
        W_avg = mean slip velocity
        d_p   = particle diameter
        mu_f  = fluid dynamic viscosity
        phi   = particle volume fraction
    """
    Re_m = q[0] * (1 - q[5]) * q[2] * q[3] / q[4]
    phi = q[5]
    f = q[1] / q[0] # rho_p/rho_f
    
    return (
        c1 + 
        c2 * phi +
        c3 * phi * Re_m +
        c4 * phi**5 * Re_m**(1/3) +
        (c5 + Re_m + c6 * phi * Re_m**2) / 
        (c7 + c8 * f)
    )
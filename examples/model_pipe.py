### Pipe flow model

from numpy import array
from math import sqrt, log10, pow, log
from scipy.optimize import bisect

# {rho,u,d,mu,eps}
dim_mat = [[ 1, 0, 0, 1, 0], # M
           [-3, 1, 1,-1, 1], # L
           [ 0,-1, 0,-1, 0]] # T

m = 5

# Parameter bound cases
Q_lo = []; Q_hi = []
Q_lo.append( array([1.0,1.0e-4,1.3,1.0e-5,1.0e-1]) ) # Laminar
Q_hi.append( array([1.4,1.0e-3,1.7,1.5e-5,1.5e-1]) )
Q_lo.append( array([1.0,0.4e-1,1.3,1.0e-5,1.0e-3]) ) # Transition
Q_hi.append( array([1.4,0.4e+0,1.7,1.5e-5,1.5e-3]) )
Q_lo.append( array([1.0,1.0e+0,1.3,1.0e-5,0.5e-1]) ) # Turbulent
Q_hi.append( array([1.4,1.0e+1,1.7,1.5e-5,2.0e-1]) )

w_u = [0,0,0,0,0]

def re_fcn(q):
    # {rho,u,d,mu,eps}
    return q[0]*q[1]*q[2]/q[3]

def f_lam(q):
    # {rho,u,d,mu,eps}
    return 64. / re_fcn(q)

def colebrook(q,f):
    # {rho,u,d,mu,eps}
    fs = sqrt(f); Re = re_fcn(q)
    return 1 + 2.*fs*log10(q[4]/3.6/q[2] + 2.51/Re/fs)

def f_tur(q):
    return bisect(lambda f: colebrook(q,f), 1e-5, 10)

Re_c = 3e3
def fcn(q):
    Re = re_fcn(q)
    if Re < Re_c:
        return f_lam(q)
    else:
        return f_tur(q)

# Error analysis functions
def dFdf(q,f):
    R = q[4]/q[2]; Re = re_fcn(q)
    return -0.5*pow(f,-1.5) + 2/log(10.) * (-1.255*pow(Re,-1)*pow(f,-1.5)) / (R/3.7 + 2.51*pow(Re,-1)*pow(f,-0.5))

if __name__ == "__main__":
    import numpy as np
    import pyutil.numeric as ut
    ## Test functions

    my_case = 2

    Q_nom = 0.5*(Q_lo[my_case]+Q_hi[my_case])
    Re_nom = re_fcn(Q_nom)
    Ep_nom = Q_nom[4]/Q_nom[2]

    f = fcn(Q_nom)

    ## Probe bisection error
    # q_s = Q_nom
    n = int(1e4)
    Q_mc = ut.qmc_unif(n,5,seed=0) * (Q_hi[my_case]-Q_lo[my_case]) + Q_lo[my_case]

    Del_f = [0] * n
    Del_F = [0] * n
    Flag  = ['unknown'] * n
    for ind in range(n):
        q_s = Q_mc[ind]
        # Iterative solver
        res = bisect(lambda f: colebrook(q_s,f), 1e-5, 10, full_output=True, disp=True)
        # Error in root equation
        f_star = res[0]
        del_F  = colebrook(q_s,f_star)
        flag   = res[1].flag
        # Linear approximation for QoI error
        diff   = dFdf(q_s,f_star)
        del_f  = del_F / diff
        # Store results
        Del_f[ind] = del_f
        Del_F[ind] = del_F
        Flag[ind]  = flag
    # Average results
    del_f_avg = np.mean(Del_f)
    del_F_avg = np.mean(Del_F)
    diverged_count = sum([ x!='converged' for x in Flag ])

    print("my_case = {}".format(my_case))
    print("")
    print("Error in F: min, avg, max = {0:+1.3e} {1:+1.3e} {2:+1.3e}".format(min(Del_F),del_F_avg,max(Del_F)))
    print("Error in f: min, avg, max = {0:+1.3e} {1:+1.3e} {2:+1.3e}".format(min(Del_f),del_f_avg,max(Del_f)))
    print("diverged_count = {}".format(diverged_count))

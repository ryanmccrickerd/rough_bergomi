import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def g(x, a):
    """
    TBSS kernel applicable to the rBergomi variance process.
    """
    return x**a

def b(k, a):
    """
    Optimal discretisation of TBSS process for minimising hybrid scheme error.
    """
    return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)

def cov(a, n):
    """
    Covariance matrix for given alpha and n, assuming kappa = 1 for
    tractability.
    """
    cov = np.array([[0.,0.],[0.,0.]])
    cov[0,0] = 1./n
    cov[0,1] = 1./((1.*a+1) * n**(1.*a+1))
    cov[1,1] = 1./((2.*a+1) * n**(2.*a+1))
    cov[1,0] = cov[0,1]
    return cov

def bs(F, K, V):
    """
    Returns the Black call price for given forward, strike and integrated
    variance.
    """
    if V == 0:
        P = np.maximum(F - K, 0)
    else:
        sv = np.sqrt(V)
        d1 = np.log(F/K) / sv + 0.5 * sv
        d2 = d1 - sv
        P = F * norm.cdf(d1) - K * norm.cdf(d2)
    return P

def bsinv(P, F, K, t):
    """
    Returns implied Black vol from given call price, forward, strike and time
    to maturity.
    """
    if P <= np.maximum(F - K, 0):
        s = 0
    else:
        def error(s):
            return bs(F, K, s**2 * t) - P
        s = brentq(error, 1e-9, 1e+9)
    return s

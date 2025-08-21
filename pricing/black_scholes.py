from math import log, sqrt, exp, pi
from scipy.stats import norm
from scipy.optimize import brentq

# -------- vanilla prices --------
def bs_call(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)
    d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# -------- greeks (call) --------
def greeks_call(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0, 0, 0, 0, 0
    d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    pdf = (1.0/sqrt(2.0*pi))*exp(-0.5*d1*d1)
    delta = norm.cdf(d1)
    gamma = pdf/(S*sigma*sqrt(T))
    vega  = S*pdf*sqrt(T)           # per 1.0 change in vol (not %)
    theta = -(S*pdf*sigma)/(2*sqrt(T)) - r*K*exp(-r*T)*norm.cdf(d2)
    rho   = K*T*exp(-r*T)*norm.cdf(d2)
    return delta, gamma, vega, theta, rho

# -------- implied vol --------
def implied_vol_call(mkt_price, S, K, T, r, low=1e-4, high=5.0):
    def f(sig): return bs_call(S, K, T, r, sig) - mkt_price
    return brentq(f, low, high)

def implied_vol_put(mkt_price, S, K, T, r, low=1e-4, high=5.0):
    def f(sig): return bs_put(S, K, T, r, sig) - mkt_price
    return brentq(f, low, high)
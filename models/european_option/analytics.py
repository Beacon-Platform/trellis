# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins

"""Black-Scholes analytical solutions for european options.

See https://en.wikipedia.org/wiki/Black–Scholes_model
"""

import numpy as np
from scipy.stats import norm


def calc_opt_price(is_call, spot, strike, texp, vol, rd, rf):
    """Calculates option price
    
    Parameters
    ----------
    is_call : bool
        True if the option is a call option, else False if a put option
    spot : float or :obj:`numpy.array`
        Current spot price(s) (S_t)
    strike : float
        Strike price (K)
    texp : float
        Time to maturity (in years) (T - t)
    vol : float
        Volatility of returns of the underlying asset (σ)
    rd : float
        Risk free rate (r)
    rf : float
        Dividend yield (q)
    
    Returns
    -------
    float
        Price
    """

    if vol <= 0 or texp <= 0 or strike <= 0:
        # Return intrinsic value
        int_val = spot * np.exp(-rf * texp) - strike * np.exp(-rd * texp)
        sign = 1 if is_call else -1
        return sign * np.maximum(int_val, 0)

    # Otherwise calculate the standard value
    d1 = calc_d1(spot, strike, texp, vol, rd, rf)
    d2 = d1 - vol * np.sqrt(texp)

    if is_call:
        return spot * np.exp(-rf * texp) * norm.cdf(d1) - strike * np.exp(-rd * texp) * norm.cdf(d2)
    else:
        return strike * np.exp(-rd * texp) * norm.cdf(-d2) - spot * np.exp(-rf * texp) * norm.cdf(-d1)


def calc_opt_delta(is_call, spot, strike, texp, vol, rd, rf):
    """Calculates option delta
    
    Delta is the partial derivative of option price with respect to the spot price of the underlying
    asset (∂C/∂S).
    
    Parameters
    ----------
    is_call : bool
        True if the option is a call option, else False if a put option
    spot : float or :obj:`numpy.array`
        Current spot price(s) (S_t)
    strike : float
        Strike price (K)
    texp : float
        Time to maturity (in years) (T - t)
    vol : float
        Volatility of returns of the underlying asset (σ)
    rd : float
        Risk free rate (r)
    rf : float
        Dividend yield (q)
    
    Returns
    -------
    float
        Delta
    """

    if vol <= 0 or texp <= 0:
        # Return intrinsic delta
        int_val = spot * np.exp(-rf * texp) - strike * np.exp(-rd * texp)

        if not is_call:
            int_val *= -1

        sign = 1 if is_call else -1
        return np.where(int_val < 0, 0, sign * np.exp(-rf * texp))

    # Otherwise calculate the standard value
    if is_call:
        return np.exp(-rf * texp) * norm.cdf(calc_d1(spot, strike, texp, vol, rd, rf))
    else:
        return -np.exp(-rf * texp) * norm.cdf(-calc_d1(spot, strike, texp, vol, rd, rf))


def calc_d1(spot, strike, texp, vol, rd, rf):
    """Calculates the d_1 value in the Black-Scholes formula with continuous yield dividends
    
    Parameters
    ----------
    spot : float or :obj:`numpy.array`
        Current spot price(s) (S_t)
    strike : float
        Strike price (K)
    texp : float
        Time to maturity (in years) (T - t)
    vol : float
        Volatility of returns of the underlying asset (σ)
    rd : float
        Risk free rate (r)
    rf : float
        Dividend yield (q)
    
    Returns
    -------
    float
        The value of d_1 for the given argument values
    """
    return (np.log(spot / strike) + (rd - rf + vol * vol / 2.0) * texp) / vol / np.sqrt(texp)

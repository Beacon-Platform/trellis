"""
Copyright: |
    Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
    Unauthorized copying of this file, via any medium, is strictly prohibited.
    Proprietary and confidential.
Product: Standard
Authors: Mark Higgins, Ben Pryke
Description: |
    Black-Scholes helper functions for deep hedging.
    
    See https://en.wikipedia.org/wiki/Black–Scholes_model
"""

import math

from scipy.stats import norm


def opt_price(is_call, spot, strike, texp, vol, rd, rf):
    """Calculates option price
    
    Parameters
    ----------
    is_call : bool
        True if the option is a call option, else False if a put option
    spot : float
        Current spot price (S_t)
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
        # return intrinsic value
        int_val = spot * math.exp(-rf * texp) - strike * math.exp(-rd * texp)
        
        if not is_call: 
            int_val *= -1
        
        return max(int_val, 0)
    
    # otherwise calculate the standard value
    d1 = calc_d1(spot, strike, texp, vol, rd, rf)
    d2 = d1 - vol * math.sqrt(texp)
    
    if is_call:
        return spot * math.exp(-rf * texp) * norm.cdf(d1) - strike * math.exp(-rd * texp) * norm.cdf(d2)
    else:
        return strike * math.exp(-rd * texp) * norm.cdf(-d2) - spot * math.exp(-rf * texp) * norm.cdf(-d1)


def opt_delta(is_call, spot, strike, texp, vol, rd, rf):
    """Calculates option delta
    
    Delta is the partial derivative of option price with respect to the spot price of the underlying
    asset (∂C/∂S).
    
    Parameters
    ----------
    is_call : bool
        True if the option is a call option, else False if a put option
    spot : float
        Current spot price (S_t)
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
        # return intrinsic delta
        int_val = spot * math.exp(-rf * texp) - strike * math.exp(-rd * texp)
        
        if not is_call:
            int_val *= -1
        
        if int_val < 0:
            return 0
        elif is_call:
            return math.exp(-rf * texp)
        else:
            return -math.exp(-rf * texp)
    
    # otherwise calculate the standard value
    if is_call:
        return math.exp(-rf * texp) * norm.cdf(calc_d1(spot, strike, texp, vol, rd, rf))
    else:
        return -math.exp(-rf * texp) * norm.cdf(-calc_d1(spot, strike, texp, vol, rd, rf))


def calc_d1(spot, strike, texp, vol, rd, rf):
    """Calculates the d_1 value in the Black-Scholes formula with continuous yield dividends
    
    Parameters
    ----------
    spot : float
        Current spot price (S_t)
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
    return (math.log(spot / strike) + (rd - rf + vol * vol / 2.) * texp) / vol / math.sqrt(texp)

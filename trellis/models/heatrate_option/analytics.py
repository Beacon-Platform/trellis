# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins, Amine Benchrifa

"""Black-Scholes analytical solutions for heat rate options.

"""

import numpy as np
from scipy.stats import norm


def calc_opt_price(is_call, spot_power, spot_gas, strike, H, texp, vol_P, vol_G, rd, rho):
    """Calculates option price of a heat rate option

    Parameters
    ----------
    is_call : bool
        True if the option is a call option, else False if a put option
    spot_power : float or :obj:`numpy.array`
        Current power price(s)
    spot_gas : float or :obj:`numpy.array`
        Current gas price(s)
    strike : float
        Strike price (K)
    H: float
        Heat rate (H)
    texp : float
        Time to maturity (in years) (T - t)
    vol_P : float
        Volatility of returns of the power asset (σ_P)
    vol_G : float
        Volatility of returns of the gas asset (σ_P)
    rd : float
        Risk free rate (r)
    rho: float
        Correlation between the power and gas prices


    Returns
    -------
    float
        Price
    """

    if vol_P <= 0 or vol_G <= 0 or texp <= 0 or strike <= 0:
        # Return intrinsic value
        int_val = (spot_power - (H * spot_gas * strike)) * np.exp(-rd * texp)
        sign = 1 if is_call else -1
        return sign * np.maximum(int_val, 0)

    vol = np.sqrt(
        vol_P ** 2 + (vol_G * (spot_gas / (H * spot_gas + strike))) ** 2 - 2 * rho * vol_P * vol_G * (spot_gas / (H * spot_gas + strike))
    )

    # Otherwise calculate the standard value
    d1 = calc_d1(spot_power, spot_gas, strike, H, texp, vol_P, vol_G, rd, rho)
    d2 = d1 - vol * np.sqrt(texp)

    if is_call:
        return (spot_power * norm.cdf(d1) - (H * spot_gas + strike) * norm.cdf(d2)) * np.exp(-rd * texp)
    else:
        return -(spot_power * norm.cdf(d1) - (H * spot_gas + strike) * norm.cdf(d2)) * np.exp(-rd * texp)


def calc_opt_delta(is_call, spot_power, spot_gas, strike, H, texp, vol_P, vol_G, rd, rho):
    """Calculates two option deltas with respect to the power and gas prices

    Delta is the partial derivative of option price with respect to the spot price of the underlying
    asset (∂C/∂S).

    Parameters
    ----------
    is_call : bool
        True if the option is a call option, else False if a put option
    spot_power : float or :obj:`numpy.array`
        Current power price(s)
    spot_gas : float or :obj:`numpy.array`
        Current gas price(s)
    strike : float
        Strike price (K)
    H: float
        Heat rate (H)
    texp : float
        Time to maturity (in years) (T - t)
    vol_P : float
        Volatility of returns of the power asset (σ_P)
    vol_G : float
        Volatility of returns of the gas asset (σ_P)
    rd : float
        Risk free rate (r)
    rho: float
        Correlation between the power and gas prices


    Returns
    -------
    pair
        (Delta_power,Delta_gas)
    """

    if vol_P <= 0 or vol_G <= 0 or texp <= 0:
        # Return intrinsic delta
        int_val = (spot_power - (strike + H * spot_gas)) * np.exp(-rd * texp)

        if not is_call:
            int_val *= -1

        sign = 1 if is_call else -1
        return np.where(int_val < 0, 0, sign)

    vol = np.sqrt(
        vol_P ** 2 + (vol_G * (spot_gas / (H * spot_gas + strike))) ** 2 - 2 * rho * vol_P * vol_G * (spot_gas / (H * spot_gas + strike))
    )

    d1 = calc_d1(spot_power, spot_gas, strike, H, texp, vol_P, vol_G, rd, rho)
    d2 = d1 - vol * np.sqrt(texp)

    # Otherwise calculate the standard value
    if is_call:

        delta_power = np.exp(-rd * texp) * norm.cdf(d1)
        delta_gas = -np.exp(-rd * texp) * H * norm.cdf(d2)

    else:
        delta_power = -np.exp(-rd * texp) * norm.cdf(d1)
        delta_gas = np.exp(-rd * texp) * H * norm.cdf(d2)

    return delta_power, delta_gas


def calc_d1(spot_power, spot_gas, strike, H, texp, vol_P, vol_G, rd, rho):
    """Calculates the d_1 value in the Black-Scholes formula with continuous yield dividends

    Parameters
    ----------
    spot_power : float or :obj:`numpy.array`
        Current power price(s)
    spot_gas : float or :obj:`numpy.array`
        Current gas price(s)
    strike : float
        Strike price (K)
    H: float
        Heat rate (H)
    texp : float
        Time to maturity (in years) (T - t)
    vol_P : float
        Volatility of returns of the power asset (σ_P)
    vol_G : float
        Volatility of returns of the gas asset (σ_P)
    rd : float
        Risk free rate (r)
    rho: float
        Correlation between the power and gas prices


    Returns
    -------
    float
        The value of d_1 for the given argument values
    """

    vol = np.sqrt(
        vol_P ** 2 + (vol_G * (spot_gas / (H * spot_gas + strike))) ** 2 - 2 * rho * vol_P * vol_G * (spot_gas / (H * spot_gas + strike))
    )

    return (np.log(spot_power / (H * spot_gas + strike)) + (vol * vol / 2.0) * texp) / (vol * np.sqrt(texp))

# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins

"""Black-Scholes-based analytical Variable Annuity calculations."""

import numpy as np
import scipy

from trellis.models.european_option import analytics as option_analytics


def calc_fair_fee(texp, gmdb, S0, vol, lam):
    """Fair fee that the insurer receives to make the whole structure zero cost"""

    def port_value(est_fee):
        def integ(t):
            fwd = S0 * np.exp(-est_fee * t)
            opt = option_analytics.calc_opt_price(False, fwd, gmdb, t, vol, 0, 0)

            return opt * np.exp(-lam * t) * lam

        drift_rn = -est_fee - lam

        val = -scipy.integrate.quad(integ, 0, texp)[0]
        if drift_rn == 0:
            val += est_fee * texp
        else:
            val += est_fee * (np.exp(drift_rn * texp) - 1) / drift_rn

        return val

    fair_fee = scipy.optimize.newton(port_value, 1e-3)
    return float(fair_fee)


def calc_delta(texp, start_time, lam, vol, fee, gmdb, account, spot):
    """Delta to put on against the VA portfolio"""

    t_fwd = texp - start_time
    n_int = max(2, int(round(0.4 * t_fwd)))
    dt_int = t_fwd / n_int
    deltas = -fee * account * np.exp(-lam * start_time) / spot * (1 - np.exp(-(fee + lam) * t_fwd)) / (fee + lam)
    for j in range(n_int):
        t_int = (j + 0.5) * dt_int  # time from current time to put expiration
        adj_strikes = gmdb * spot / account * np.exp(fee * t_int)
        d1s = (np.log(spot / adj_strikes) + vol * vol * t_int / 2) / vol / np.sqrt(t_int)
        put_deltas = scipy.stats.norm.cdf(d1s * -1) * -1 * account / spot * np.exp(-fee * t_int)
        t_s = j * dt_int
        t_e = t_s + dt_int
        deltas += (np.exp(-lam * t_s) - np.exp(-lam * t_e)) * put_deltas * np.exp(-lam * start_time)

    return deltas

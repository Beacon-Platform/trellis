# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins

"""European option deep hedging and Black-Scholes models."""

from trellis.models.european_option.analytics import calc_opt_delta, calc_opt_price
from trellis.models.european_option.model import EuropeanOption, Hyperparams

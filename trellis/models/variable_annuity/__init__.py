# Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
# License: MIT
# Authors: Benjamin Pryke, Mark Higgins

"""Variable annuity deep hedging and Black-Scholes models."""

from trellis.models.variable_annuity.analytics import calc_delta, calc_fair_fee
from trellis.models.variable_annuity.model import Hyperparams, VariableAnnuity

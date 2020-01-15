"""
Copyright: |
    Copyright (C) 2020 Beacon Platform Inc. - All Rights Reserved.
    Unauthorized copying of this file, via any medium, is strictly prohibited.
    Proprietary and confidential.
Product: Standard
Authors: Mark Higgins, Ben Pryke
Description: Utilities for the deep hedging scripts.
"""

import time


def get_duration_since_desc(start):
    """Returns a string {min}:{sec} describing the duration since `start`
    
    Parameters
    ----------
    start : int or float
        Timestamp
    """
    
    end = time.time()
    duration = round(end - start)
    minutes, seconds = divmod(duration, 60)
    return '{:02d}:{:02d}'.format(minutes, seconds)

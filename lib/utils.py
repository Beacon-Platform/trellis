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

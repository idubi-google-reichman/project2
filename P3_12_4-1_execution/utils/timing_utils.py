# import timeit
import time
from utils.logging_utils import LoggerUtility, LOG_LEVEL

# from functools import partial


# Function to capture the result and measure the time
def capture_and_time(func, *args, **kwargs):
    # Capture the start time
    start_time = time.time()
    # Capture the result
    result = func(*args, **kwargs)
    # Capture the end time
    end_time = time.time()
    # calculate execution time
    execution_time = end_time - start_time
    if result is not None:
        return result, execution_time
    else:
        return execution_time


def convert_count_time_to_human(elapsed_time):
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = (seconds - int(seconds)) * 1000

    return f"{int(hours)} (H):{int(minutes)} (Mi):{int(seconds)} (Sec):{milliseconds:.0f} (ms)"

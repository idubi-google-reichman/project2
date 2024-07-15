# import timeit
import time

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
    return result, execution_time

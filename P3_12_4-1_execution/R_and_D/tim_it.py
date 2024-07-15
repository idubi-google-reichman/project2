import timeit


# Define a function to be timed
def example_function(params={"repeats": 1000}):

    params["result"] = sum([i for i in range(params["repeats"])])
    return params["result"]


# Measure execution time
a = {"repeats": 10}
execution_time = timeit.timeit(lambda: example_function(a), number=700)
print(f" execution result : {a['result']} Execution time: {execution_time} seconds")

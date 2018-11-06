import time


def meassureFncRuntime(fnc):
    start = time.time()
    output = fnc()
    end = time.time()
    ellpsed_time = end - start
    return ellpsed_time, output
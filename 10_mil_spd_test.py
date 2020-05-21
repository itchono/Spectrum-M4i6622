import numpy as np
from numba import jit
import time
import matplot

def calculateMatrix(function0, function1, function2, function3):
    # uses matrix multiplcation to map the functions from a form like [1, 2, 3, 4]
    # into [1, 0, 0, 0, 2, 0, 0, 0 ...]
    try:
        ELEMENT_SIZE = 10000000
        QUARTER_SIZE = int(ELEMENT_SIZE/4)

        values = np.arange(0,QUARTER_SIZE, 1) # creates array of 2.5 million elements

        arr = [function0(values), function1(values), function2(values), function3(values)]

        tsfm = np.zeros((ELEMENT_SIZE, QUARTER_SIZE))
        # not doable
        # needs 1.82 TiB of RAM LOL

        # idea was to map from Quarterx1 vector to fullsizex1 vector
        tsfm[0::4,:] = np.identity(QUARTER_SIZE)

        for i in range(len(arr)):
            arr[i] = np.matmul(tsfm, np.transpose(arr[i]))
            tsfm = np.roll(tsfm, 1, axis=0) # roll everything down by one space

        total = arr[0] + arr[1] + arr[2] + arr[3]
        # takes 0.10 seconds for just 20 elements lol
    except:
        print("Failed to allocate memory")

def matrixFlatCat(function0, function1, function2, function3):
    # takes columns of matrix as the 4 functions, concatenates, and flattens them, therefore stacking them.
    ELEMENT_SIZE = 10000000
    QUARTER_SIZE = int(ELEMENT_SIZE/4)

    values = np.arange(0,QUARTER_SIZE, 1) # creates array of 2.5 million elements

    # evaluation of functions
    arr0 = function0(values)
    arr1 = function1(values)
    arr2 = function2(values)
    arr3 = function3(values)

    big = np.column_stack((arr0, arr1, arr2, arr3))
    return big.flatten()
    # around 0.5 seconds

def matrixFlatCatVariant(function0, function1, function2, function3):
    # takes columns of matrix as the 4 functions, concatenates, and flattens them, therefore stacking them.
    ELEMENT_SIZE = 10000000
    QUARTER_SIZE = int(ELEMENT_SIZE/4)

    values = np.arange(0,QUARTER_SIZE, 1) # creates array of 2.5 million elements

    # evaluation of functions
    arr0 = function0(values)
    arr1 = function1(values)
    arr2 = function2(values)
    arr3 = function3(values)

    big = np.hstack((arr0, arr1, arr2, arr3)).T
    return big.flatten()
    # around 0.5 seconds

def matrixFlatDat(function0, function1, function2, function3):
    # creates a matrix directly with 4 function as columns, and flattens them, therefore stacking them.
    ELEMENT_SIZE = 10000000
    QUARTER_SIZE = int(ELEMENT_SIZE/4)

    values = np.arange(0,QUARTER_SIZE, 1) # creates array of 2.5 million elements

    big = np.zeros((QUARTER_SIZE, 4))

    big[:,0] = function0(values)
    big[:,1] = function1(values)
    big[:,2] = function2(values)
    big[:,3] = function3(values)
    # evaluation of functions

    return big.flatten()
    # around 0.5 seconds

def baseline(function0, function1, function2, function3):

    ELEMENT_SIZE = 10000000
    QUARTER_SIZE = int(ELEMENT_SIZE/4)

    buffer = [0] * (ELEMENT_SIZE)

    for i in range(QUARTER_SIZE):
        buffer[4*i] = (function0(i))
        buffer[4*i+1] = (function1(i))
        buffer[4*i+2] = (function2(i))
        buffer[4*i+3] = (function3(i))

    return buffer
    # around 51.7 seconds

@jit(nopython=True)
def numbaAccelerate():

    def function0(x):
        return np.sin(np.pi*x)

    def function1(x):
        return np.cos(np.pi*x)

    def function2(x):
        return function0(x) + function1(x)

    def function3(x):
        return function0(x) * function1(x)

    ELEMENT_SIZE = 10000000
    QUARTER_SIZE = int(ELEMENT_SIZE/4)

    buffer = [0] * (ELEMENT_SIZE)

    for i in range(QUARTER_SIZE):
        buffer[4*i] = (function0(i))
        buffer[4*i+1] = (function1(i))
        buffer[4*i+2] = (function2(i))
        buffer[4*i+3] = (function3(i))

    return buffer
    # around 1.27s

@jit(nopython=True, parallel=True)
def numbaFlatCat():
    def function0(x):
        return np.sin(np.pi*x)

    def function1(x):
        return np.cos(np.pi*x)

    def function2(x):
        return function0(x) + function1(x)

    def function3(x):
        return function0(x) * function1(x)
    ELEMENT_SIZE = 10000000
    QUARTER_SIZE = int(ELEMENT_SIZE/4)

    values = np.arange(0,QUARTER_SIZE, 1) # creates array of 2.5 million elements

    # evaluation of functions
    arr0 = function0(values)
    arr1 = function1(values)
    arr2 = function2(values)
    arr3 = function3(values)

    big = np.hstack((arr0, arr1, arr2, arr3)).T
    return big.flatten()

def f0(x):
    return np.sin(np.pi*x)

def f1(x):
    return np.cos(np.pi*x)

def f2(x):
    return f0(x) + f1(x)

def f3(x):
    return f0(x) * f1(x)

def f0t(x):
    return 0*x

def f1t(x):
    return 1*x

def f2t(x):
    return 2*x

def f3t(x):
    return 3*x


if __name__ == "__main__":

    times = {f.__name__:[] for f in [calculateMatrix, matrixFlatCat, matrixFlatCatVariant, matrixFlatDat, numbaAccelerate, numbaFlatCat]}

    for fnc in [calculateMatrix, matrixFlatCat, matrixFlatCatVariant, matrixFlatDat]:
        print("Function name: {}".format(fnc.__name__))
        t_start = time.perf_counter_ns()

        m = fnc(f0, f1, f2, f3)

        t_end = time.perf_counter_ns()-t_start

        print("Time taken: {} ns".format(t_end))

    for fnc in [numbaAccelerate, numbaFlatCat]:
        print("Function name: {}".format(fnc.__name__))
        t_start = time.perf_counter_ns()

        m = fnc()

        print("Time taken: {} ns".format(time.perf_counter_ns()-t_start))

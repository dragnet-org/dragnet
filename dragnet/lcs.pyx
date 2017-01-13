"""
longest common subsequence

modified from the code snippets at
http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_subsequence#Python

cython -a lcs.pyx to output HTML
"""
import numpy as np

cimport cython
cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef inline int int_max(int a, int b): return a if a >= b else b


@cython.boundscheck(False)
def longest_common_subsequence(X, Y):
    """Compute and return the longest common subsequence matrix
    X, Y are list of strings"""
    cdef int m = len(X)
    cdef int n = len(Y)

    # use numpy array for memory efficiency with long sequences
    # lcs is bounded above by the minimum length of x, y
    assert min(m+1, n+1) < 65535

    #cdef np.ndarray[np.int32_t, ndim=2] C = np.zeros([m+1, n+1], dtype=np.int32)
    cdef np.ndarray[np.uint16_t, ndim=2] C = np.zeros([m+1, n+1], dtype=np.uint16)

    # convert X, Y to C++ standard containers
    cdef vector[string] xx = X
    cdef vector[string] yy = Y

    cdef int i, j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if xx[i-1] == yy[j-1]:
                C[i, j] = C[i-1, j-1] + 1
            else:
                C[i, j] = int_max(C[i, j-1], C[i-1, j])
    return C

def print_diff(X, Y):
    """Print the difference of the sequences

    Usage:
        print_diff('wow, this is the first string', 'this is the second string here')
        print_diff(['wow', 'this', 'is', 'the', 'first', 'string'],
                   ['this', 'is', 'the', 'second', 'string', 'here'])
    """
    C = longest_common_subsequence(X, Y)
    i = len(X)
    j = len(Y)
    diff = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and X[i-1] == Y[j-1]:
            #diff.append("  " + X[i-1])
            i -= 1
            j -= 1
        else:
            if j > 0 and (i == 0 or C[i][j-1] >= C[i-1][j]):
                diff.append("+ " + Y[j-1])
                j -= 1
            elif i > 0 and (j == 0 or C[i][j-1] < C[i-1][j]):
                diff.append("- " + X[i-1])
                i -= 1
    diff.reverse()
    print diff




def check_inclusion(x, y):
    """Given x, y (formatted as input to longest_common_subsequence)
    return a vector v of True/False with length(x)
    where v[i] == True if x[i] is in the longest common subsequence with y"""
    if len(y) == 0:
        return [False] * len(x)

    c = longest_common_subsequence(x, y)

    i = len(x)
    j = len(y)
    ret = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and x[i-1] == y[j-1]:
            ret.append(True)
            i -= 1
            j -= 1
        else:
            if j > 0 and (i == 0 or c[i][j-1] >= c[i-1][j]):
                j -= 1
            elif i > 0 and (j == 0 or c[i][j-1] < c[i-1][j]):
                ret.append(False)
                i -= 1

    ret.reverse()
    return ret

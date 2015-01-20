
cimport cython
cimport numpy as np
np.import_array()

import numpy as np
from scipy.ndimage import gaussian_filter


@cython.boundscheck(False)
@cython.cdivision(True)
def weninger_sx_sdx(np.ndarray[np.float64_t, ndim=1] x):
    """blocks = a 1D array of something (their content-tag-ratios, length, ..)
    Computes and returns the smoothed version of x and
    absolute smoothed discount (eqn 4)"""
    sigma = 1.0
    cdef int alpha = 3

    cdef int nx = len(x)

    # compute absolute discount
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] dx = \
        np.ascontiguousarray(np.zeros(nx), dtype=np.float64)
    cdef int k
    cdef int start, end
    for k in range(nx-1):
        start = k + 1
        end = min(k + 1 + alpha, nx)
        for j in range(start, end):
            dx[k] += x[j]
        dx[k] /= (end - start)
        dx[k] = abs(dx[k] - x[k])
    dx[nx-1] = abs(0.5 * (x[nx-1] + x[nx-2]) - x[nx-1])

    # smooth
    ret = np.zeros((nx, 2))
    ret[:, 0] = gaussian_filter(x, sigma=sigma)
    ret[:, 1] = gaussian_filter(dx, sigma=sigma)

    return ret


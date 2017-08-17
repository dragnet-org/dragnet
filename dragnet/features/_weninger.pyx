cimport cython
cimport numpy as np

import numpy as np
from scipy.ndimage import gaussian_filter

from dragnet.compat import bytes_block_list_cast, str_list_cast


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _blocks_to_ctrs(blocks):
    cdef int nblocks = len(blocks)
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] block_lengths = \
        np.empty(nblocks, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] tag_counts = \
        np.empty(nblocks, dtype=np.float64)
    cdef int i
    for i in range(0, nblocks):
        block_lengths[i] = len(blocks[i].text)
        tag_counts[i] = blocks[i].features[b'tagcount']

    return block_lengths / np.maximum(tag_counts, 1.0)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef sx_sdx(np.ndarray[np.float64_t, ndim=1] x, float sigma=1.0):
    """
    Computes and returns the smoothed values of ``x`` and its smoothed absolute
    differences (eqn 4 in paper).

    Args:
        x (`np.ndarray`): 1D array of some aspect of a sequence of blocks, e.g.
            their content-tag-ratios or lengths.
        sigma (float): Standard deviation for Gaussian kernel used in smoothing.

    Returns:
        :class:`np.ndarray`: 2D array of shape (len(x), 2), where values are
            floats corresponding to the smoothed and smoothed absolute difference
            values for each item in ``x``.

    TODO: make alpha an arg? would require un-hardcoding some logic
    """
    cdef int alpha = 3
    cdef int nx = len(x)

    # find the derivatives for each element in x:
    # subtract the element's value from the mean of the next alpha elements
    # (and take the absolute value)
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] dx = \
        np.ascontiguousarray(np.zeros(nx), dtype=np.float64)
    cdef int k
    cdef int start, end
    for k in range(0, nx - 1):
        start = k + 1
        end = min(k + 1 + alpha, nx)
        for j in range(start, end):
            dx[k] += x[j]
        dx[k] /= (end - start)
        dx[k] = abs(dx[k] - x[k])
    dx[nx - 1] = abs((0.5 * (x[nx - 1] + x[nx - 2])) - x[nx - 1])

    # smooth array values
    ret = np.empty((nx, 2), dtype=np.float64)
    ret[:, 0] = gaussian_filter(x, sigma=sigma)
    ret[:, 1] = gaussian_filter(dx, sigma=sigma)

    return ret


def make_weninger_features(blocks, sigma=1.0):
    # NOTE: These are outbound values, so they are cast to strings
    return str_list_cast(sx_sdx(_blocks_to_ctrs(bytes_block_list_cast(blocks)), sigma=sigma))

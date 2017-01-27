"""
TODO
"""
cimport cython
cimport numpy as np

import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def make_kohlschuetter_features(blocks):
    cdef int nblocks = len(blocks)
    if nblocks < 3:
        raise ValueError(
            'at least 3 blocks are needed to make Kohlschuetter features')

    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] features = \
        np.ascontiguousarray(np.empty((nblocks, 6)), dtype=np.float64)

    cdef int i = 0
    features[i, :] = (
        0.0, 0.0,
        blocks[i].link_density, blocks[i].text_density,
        blocks[i + 1].link_density, blocks[i + 1].text_density
        )
    for i in range(1, nblocks - 1):
        prevb = blocks[i - 1]
        currb = blocks[i]
        nextb = blocks[i + 1]
        features[i, :] = (
            prevb.link_density, prevb.text_density,
            currb.link_density, currb.text_density,
            nextb.link_density, nextb.text_density
            )
    i = nblocks - 1
    features[i, :] = (
        blocks[i - 1].link_density, blocks[i - 1].text_density,
        blocks[i].link_density, blocks[i].text_density,
        0.0, 0.0
        )
    return features

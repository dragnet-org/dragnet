
cimport cython
cimport numpy as np
np.import_array()

from libcpp.pair cimport pair
from libc.stdint cimport uint32_t
from libcpp.vector cimport vector
from libcpp.string cimport string

import numpy as np


cdef extern from "_readability.cc":
    cdef void _readability_features(
        vector[uint32_t] &,
        vector[vector[pair[uint32_t, int] ] ] &,
        vector[vector[uint32_t] ]  &,
        vector[string] &,
        vector[double] &,
        int &,
        double*)


def readability_features(blocks, *args, **kwargs):
    """
    Features inspired by Readability
    """
    cdef int nblocks = len(blocks)

    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] features = \
        np.ascontiguousarray(np.zeros((nblocks, 1)), dtype=np.float64)

    # elements in the blocks we'll need
    cdef vector[uint32_t] block_text_len
    cdef vector[vector[pair[uint32_t, int] ] ] block_readability_class_weights
    cdef vector[vector[uint32_t] ] block_ancestors
    cdef vector[string] block_start_tag
    cdef vector[double] block_link_density

    block_text_len.reserve(nblocks)
    block_readability_class_weights.reserve(nblocks)
    block_ancestors.reserve(nblocks)
    block_start_tag.reserve(nblocks)
    block_link_density.reserve(nblocks)

    for block in blocks:
        block_text_len.push_back(len(block.text))
        block_readability_class_weights.push_back(
            block.features['readability_class_weights'])
        block_ancestors.push_back(block.features['ancestors'])
        block_start_tag.push_back(block.features['block_start_tag'])
        block_link_density.push_back(block.link_density)

    _readability_features(
        block_text_len,
        block_readability_class_weights,
        block_ancestors,
        block_start_tag,
        block_link_density,
        nblocks,
        &features[0, 0]
    )

    return features

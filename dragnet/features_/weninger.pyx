"""
TODO
"""
cimport cython
cimport numpy as np
np.import_array()

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans

from .base import Feature


def _blocks_to_ctrs(blocks):
    block_lengths = np.array(
        [len(block.text) for block in blocks], dtype=np.float64)
    # each block's tag count must be *at least* 1.0
    tag_counts = np.maximum(
        np.array([block.features['tagcount'] for block in blocks]), 1.0)
    return block_lengths / tag_counts


class WeningerFeatures(Feature):

    def fit(self, blocks, y=None):
        return self

    def transform(self, blocks, y=None):
        """
        Computes the content to tag ratio per block and returns the smoothed
        values and the smoothed absolute differences for each block.

        Args:
            blocks (List[Block]): as output by :class:`Blockifier.blockify`
            y (None): This isn't used, it's only here for API consistency.

        Returns:
            :class:`np.ndarray`: 2D array of shape (len(x), 2), where values are
                floats corresponding to the smoothed and smoothed absolute
                difference values for each block.
        """
        return sx_sdx(_blocks_to_ctrs(blocks))


class WeningerClusteredFeatures(Feature):

    def fit(self, blocks, y=None):
        self.kmeans = KMeans(n_clusters=3, n_init=3, max_iter=50, tol=0.001)
        self.kmeans.fit(sx_sdx(_blocks_to_ctrs(blocks)))
        # set the cluster center closest to the origin to exactly (0.0, 0.0)
        self.kmeans.cluster_centers_.sort(axis=0)
        self.kmeans.cluster_centers_[0, :] = np.zeros(2)
        return self

    def transform(self, blocks, y=None):
        """
        Computes the content to tag ratio per block and returns the smoothed
        values and the smoothed absolute differences for each block.

        Args:
            blocks (List[Block]): as output by :class:`Blockifier.blockify`
            y (None): This isn't used, it's only here for API consistency.

        Returns:
            :class:`np.ndarray`: 1D array of shape (len(feature_mat), 1), where
                values are either 0 or 1, corresponding to the kmeans prediction
                of content (1) or not-content (0)
        """
        return (self.kmeans.predict(sx_sdx(_blocks_to_ctrs(blocks))) != 0).astype(int)


@cython.boundscheck(False)
@cython.cdivision(True)
def sx_sdx(np.ndarray[np.float64_t, ndim=1] x):
    """
    Computes and returns the smoothed values of ``x`` and its smoothed absolute
    differences (eqn 4 in paper).

    Args:
        x (`np.ndarray`): 1D array of some aspect of a sequence of blocks, e.g.
            their content-tag-ratios or lengths.

    Returns:
        :class:`np.ndarray`: 2D array of shape (len(x), 2), where values are
            floats corresponding to the smoothed and smoothed absolute difference
            values for each item in ``x``.

    TODO: make alpha an arg? would require un-hardcoding some logic
    """
    sigma = 1.0
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

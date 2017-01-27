import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

from ._weninger import make_weninger_features


class WeningerFeatures(BaseEstimator, TransformerMixin):
    """
    An sklearn-style transformer that takes an ordered sequence of ``Block`` objects
    and returns a 2D array of content tag-based features, as described by
    Weninger et al.

    References:
        Weninger, Tim, William H. Hsu, and Jiawei Han. "CETR: content extraction
            via tag ratios." Proceedings of the 19th international conference on
            World wide web. ACM, 2010.
    """

    __name__ = 'weninger'

    def __init__(self, sigma=1.0):
        self.sigma = 1.0

    def fit(self, blocks, y=None):
        """
        This method returns the current instance unchanged, since no fitting is
        required for this ``Feature``. It's here only for API consistency.
        """
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
        return make_weninger_features(blocks, sigma=self.sigma)


class ClusteredWeningerFeatures(BaseEstimator, TransformerMixin):
    """
    An sklearn-style transformer that takes an ordered sequence of ``Block`` objects
    and returns a 2D array of content tag-based features that have been clustered
    using a modified k-means algorithm, as described by Weninger et al.

    Args:
        n_clusters (int): Number of clusters to form and centroids to generate.
        n_init (int): Number of times the k-means algorithm will be run with
            different centroid seeds. The final results will be the best output
            of ``n_init`` consecutive runs in terms of inertia.
        max_iter (int): Max number of k-means algorithm iterations for a single run.
        tol (float): Relative tolerance wrt inertia to declare convergence.

    References:
        Weninger, Tim, William H. Hsu, and Jiawei Han. "CETR: content extraction
            via tag ratios." Proceedings of the 19th international conference on
            World wide web. ACM, 2010.
    """

    __name__ = 'clustered_weninger'

    def __init__(self, n_clusters=3, n_init=3, max_iter=50, tol=0.001):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol)

    def fit(self, blocks, y=None):
        """
        Fit a k-means clustering model using an ordered sequence of blocks.
        """
        self.kmeans.fit(make_weninger_features(blocks))
        # set the cluster center closest to the origin to exactly (0.0, 0.0)
        self.kmeans.cluster_centers_.sort(axis=0)
        self.kmeans.cluster_centers_[0, :] = np.zeros(2)
        return self

    def transform(self, blocks, y=None):
        """
        Computes the content to tag ratio per block, smooths the values, then
        predicts content (1) or not-content (0) using a fit k-means cluster model.

        Args:
            blocks (List[Block]): as output by :class:`Blockifier.blockify`
            y (None): This isn't used, it's only here for API consistency.

        Returns:
            :class:`np.ndarray`: 2D array of shape (len(feature_mat), 1), where
                values are either 0 or 1, corresponding to the kmeans prediction
                of content (1) or not-content (0).
        """
        preds = (self.kmeans.predict(make_weninger_features(blocks)) > 0).astype(int)
        return np.reshape(preds, (-1, 1))

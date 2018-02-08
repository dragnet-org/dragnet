from sklearn.base import BaseEstimator, TransformerMixin

from ._readability import make_readability_features


class ReadabilityFeatures(BaseEstimator, TransformerMixin):
    """
    An sklearn-style transformer that takes an ordered sequence of ``Block`` objects
    and returns a 2D array of subtree-based features, based on ``readability``.
    """

    __name__ = 'readability'

    def fit(self, blocks, y=None):
        """
        This method returns the current instance unchanged, since no fitting is
        required for this ``Feature``. It's here only for API consistency.
        """
        return self

    def transform(self, blocks, y=None):
        """
        Transform an ordered sequence of blocks into a 2D features matrix with
        shape (num blocks, num features).

        Args:
            blocks (List[Block]): as output by :class:`Blockifier.blockify`
            y (None): This isn't used, it's only here for API consistency.

        Returns:
            `np.ndarray`: 2D array of shape (num blocks, 1)
        """
        return make_readability_features(blocks)

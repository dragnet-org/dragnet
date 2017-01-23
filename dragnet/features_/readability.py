from sklearn.base import BaseEstimator, TransformerMixin

from ._readability import make_readability_features


class ReadabilityFeatures(BaseEstimator, TransformerMixin):
    """
    TODO
    """
    _name = 'readability'

    def fit(self, blocks, y=None):
        """
        TODO
        """
        return self

    def transform(self, blocks, y=None):
        """
        Args:
            blocks (List[Block]): as output by :class:`Blockifier.blockify`
            y (None): This isn't used, it's only here for API consistency.

        Returns:
            `np.ndarray`: 2D array of shape (num blocks, 1)
        """
        return make_readability_features(blocks)

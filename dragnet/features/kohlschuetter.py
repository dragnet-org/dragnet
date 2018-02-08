# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ._kohlschuetter import make_kohlschuetter_features


class KohlschuetterFeatures(BaseEstimator, TransformerMixin):
    """
    An sklearn-style transformer that takes an ordered sequence of ``Block`` objects
    and returns a 2D array of text and link density-based features, as described
    by Kohlschütter et al.

    References:
        http://www.l3s.de/~kohlschuetter/publications/wsdm187-kohlschuetter.pdf
    """

    __name__ = 'kohlschuetter'

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
            `np.ndarray`: 2D array of shape (num blocks, 6), where values are
                floats corresponding to the link and text densities of
                a block and its immediate neighbors in the sequence.
        """
        return make_kohlschuetter_features(blocks)

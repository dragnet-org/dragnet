# -*- coding: utf-8 -*-
"""
A *rough* implementation of that described by Kohlschütter et al.:
   http://www.l3s.de/~kohlschuetter/publications/wsdm187-kohlschuetter.pdf
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ._kohlschuetter import make_kohlschuetter_features


class KohlschuetterFeatures(BaseEstimator, TransformerMixin):
    """
    The text density/link density features
    from Kohlschuetter. Implements the features interface.

    TODO: better docs
    """
    __name__ = 'kohlschuetter'

    def fit(self, blocks, y=None):
        return self

    def transform(self, blocks, y=None):
        """
        Args:
            blocks (List[Block]): as output by :class:`Blockifier.blockify`
            y (None): This isn't used, it's only here for API consistency.

        Returns:
            `np.ndarray`: 2D array of shape (num blocks, 6), where values are
                floats corresponding to the link and text densities of
                a block and its immediate neighbors in the sequence.
        """
        return make_kohlschuetter_features(blocks)

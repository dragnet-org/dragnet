import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CSSFeatures(BaseEstimator, TransformerMixin):
    """
    An sklearn-style transformer that takes an ordered sequence of ``Block`` objects
    and returns a 2D array of CSS-based features, where each value is 0 or 1,
    depending on the absence or presence of certain tokens in a block's
    CSS id or class attribute.
    """
    __name__ = 'css'

    # tokens that we search for in each block's CSS attribute
    # first 'id', then 'class'
    attribute_tokens = (
        ('id',
         ('nav', 'ss', 'top', 'content', 'link', 'title', 'comment', 'tools',
          'rating', 'ss')
         ),
        ('class',
         ('menu', 'widget', 'nav', 'share', 'facebook', 'cat', 'top', 'content',
          'item', 'twitter', 'button', 'title', 'header', 'ss', 'post',
          'comment', 'meta', 'alt', 'time', 'depth', 'thread', 'author', 'tools',
          'reply', 'url', 'avatar')
         )
        )

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
            `np.ndarray`: 2D array of shape (num blocks, num CSS attributes),
                where values are either 0 or 1, indicating the absence or
                presence of a given token in a CSS attribute on a given block.
        """
        feature_vecs = (
            tuple(re.search(token, block.css[attrib]) is not None
                  for block in blocks)
            for attrib, tokens in self.attribute_tokens
            for token in tokens
            )
        return np.column_stack(tuple(feature_vecs)).astype(int)

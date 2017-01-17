import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from dragnet.util import sliding_window


class Feature(BaseEstimator, TransformerMixin):
    """
    Abstract Base Class for all classes of ``Feature``.
    """

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X, y=None):
        raise NotImplementedError


class StandardizedFeature(BaseEstimator, TransformerMixin):
    """
    Args:
        feature (:class:`Feature`)
        scaler (:class:`StandardScaler`): default None
        with_mean (bool): If True, center the ``blocks`` data before scaling.
        with_std (bool): If True, scale the ``blocks`` data to unit variance
            (or, equivalently, unit standard deviation).
        copy (bool): If False, try to avoid a copy and do in-place scaling.
    """

    def __init__(self, feature, scaler=None,
                 with_mean=True, with_std=True, copy=True):
        self.feature = feature
        self.scaler = scaler
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def fit(self, blocks, y=None):
        """
        Args:
            blocks (List[Block]): as output by :class:`Blockifier.blockify`
            y (None): This isn't used, it's only here for API consistency.

        Returns:
            :class:`StandardizedFeature`: an instance of this class with the
                ``self.scaler`` attribute fit to the ``blocks`` data

        Note:
            When fitting the :class:`StandardScaler` object, you'll probably
                want to determine the mean and/or std of *multiple* HTML files'
                blocks, rather than just a single observation. To do that, just
                concatenate all of the blocks together in a single iterable.

            In contrast, you'll typically apply :meth:`transform` to a *single*
                HTML file's blocks at a time.
        """
        if self.scaler is None:
            feature_array = self.feature.transform(blocks)
            self.scaler = StandardScaler(
                copy=self.copy, with_mean=self.with_mean, with_std=self.with_std
                ).fit(feature_array)
        return self

    def transform(self, blocks, y=None):
        """
        Args:
            blocks (List[Block]): as output by :class:`Blockifier.blockify`
            y (None): This isn't used, it's only here for API consistency.

        Returns:
            `np.ndarray`: 2D array of shape (num blocks, num sub-features),
                where ``blocks`` data has been transformed by ``self.feature``
                and optionally standardized by ``self.scaler``.
        """
        if self.scaler is None:
            return self.feature.transform(blocks)
        else:
            return self.scaler.transform(self.feature.transform(blocks))


class CSSFeatures(Feature):
    """
    Class of features from id/class attributes.

    The features are 0/1 flags whether the attributes have
    a give set of tokens

    TODO: better documentation
    """

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
        required for this :class:`Feature`. It's here only for API consistency.
        """
        return self

    def transform(self, blocks, y=None):
        """
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


class KohlschuetterFeatures(Feature):
    """
    The text density/link density features
    from Kohlschuetter. Implements the features interface.

    TODO: better docs
    """

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
        len_blocks = len(blocks)
        if len_blocks < 3:
            raise ValueError(
                'at least 3 blocks are needed to make Kohlschuetter features')

        features = np.empty((len_blocks, 6), dtype=float)
        i = 0
        features[i, :] = (
            0.0, 0.0,
            blocks[i].link_density, blocks[i].text_density,
            blocks[i + 1].link_density, blocks[i + 1].text_density
            )
        for i, (prevb, currb, nextb) in enumerate(sliding_window(blocks, 3)):
            features[i + 1, :] = (
                prevb.link_density, prevb.text_density,
                currb.link_density, currb.text_density,
                nextb.link_density, nextb.text_density
                )
        i = len_blocks - 1
        features[i, :] = (
            blocks[i - 1].link_density, blocks[i - 1].text_density,
            blocks[i].link_density, blocks[i].text_density,
            0.0, 0.0
            )
        return features

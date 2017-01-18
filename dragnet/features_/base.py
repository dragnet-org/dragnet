"""
TODO: docs
TODO: are ABCs *necessary*?
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


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

    def fit(self, blocks, y=None, force=False):
        """
        Args:
            blocks (List[Block]): as output by :class:`Blockifier.blockify`
            y (None): This isn't used, it's only here for API consistency.
            force (bool): If True, *always* fit ``self.scaler``, even if it has
                already been fit. If False, *only* fit ``self.scaler`` if it has
                not yet been fit.

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
        if force is True or self.scaler is None:
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

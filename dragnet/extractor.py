import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier

from .compat import string_
from .util import get_and_union_features


class Extractor(BaseEstimator, ClassifierMixin):
    """
    An sklearn-style classifier that extracts the main content (and/or comments)
    from an HTML document.

    Args:
        blockifier (``Blockifier``)
        features (str or List[str], ``Features`` or List[``Features``], or List[Tuple[str, ``Features``]]):
            One or more features to be used to transform blocks into a matrix of
            numeric values. If more than one, a :class:`FeatureUnion` is
            automatically constructed. See :func:`get_and_union_features`.
        model (:class:`ClassifierMixin`): A scikit-learn classifier that takes
             a numeric matrix of features and outputs a binary prediction of
            1 for content or 0 for not-content. If None, a :class:`ExtraTreesClassifier`
            with default parameters is used.
        to_extract (str or Sequence[str]): Type of information to extract from
            an HTML document: 'content', 'comments', or both via ['content', 'comments'].
        prob_threshold (float): Minimum prediction probability of a block being
            classified as "content" for it actually be taken as such.
        max_block_weight (int): Maximum weight that a single block may be given
            when training the extractor model, where weights are set equal to
            the number of tokens in each block.

    Note:
        If ``prob_threshold`` is not None, then ``model`` must implement the
            ``predict_proba()`` method.
    """

    def __init__(self, blockifier,
                 features=('kohlschuetter', 'weninger', 'readability'),
                 model=None,
                 to_extract='content', prob_threshold=0.5, max_block_weight=200):
        self.blockifier = blockifier
        self.features = features
        # initialize model
        if model is None:
            self.model = ExtraTreesClassifier()
        elif isinstance(model, ClassifierMixin):
            self.model = model
        else:
            raise TypeError('invalid `model` type: "{}"'.format(type(model)))
        if isinstance(to_extract, string_):
            self.to_extract = (to_extract,)
        else:
            self.to_extract = tuple(to_extract)
        self.prob_threshold = prob_threshold
        self.max_block_weight = max_block_weight
        self._positive_idx = None

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, feats):
        self._features = get_and_union_features(feats)

    def fit(self, blocks, labels, weights=None):
        """
        Fit :class`Extractor` features and model to a training dataset.

        Args:
            blocks (List[Block])
            labels (``np.ndarray``)
            weights (``np.ndarray``)

        Returns:
            :class`Extractor`
        """
        features_mat = self.features.fit_transform(blocks)
        if weights is None:
            self.model.fit(features_mat, labels)
        else:
            self.model.fit(features_mat, labels, sample_weight=weights)
        return self

    def concatenate_data(self, data):
        """
        Concatenate the blocks, labels, and weights of many files' data.
        Primarily useful for training/testing an :class`Extractor`.

        Args:
            data: Output of :func:`dragnet.data_processing.prepare_all_data`.

        Returns:
            Tuple[List[Block], np.array(int), np.array(int)]: All blocks, all
                labels, and all weights, respectively.
        """
        all_blocks = []
        all_labels = np.empty(0, dtype=int)
        all_weights = np.empty(0, dtype=int)
        for html, content, comments in data:
            blocks = self.blockifier.blockify(html)
            if not self._has_enough_blocks(blocks):
                continue
            all_blocks.extend(blocks)
            labels, weights, _ = self._get_labels_and_weights(
                content, comments)
            all_labels = np.hstack((all_labels, labels))
            all_weights = np.hstack((all_weights, weights))
        return all_blocks, all_labels, all_weights

    def _has_enough_blocks(self, blocks):
        if len(blocks) < 3:
            logging.warning(
                'extraction failed: too few blocks (%s)', len(blocks))
            return False
        return True

    def _get_labels_and_weights(self, content, comments):
        """
        Args:
            content (Tuple[np.array[int], np.array[int], List[str]])
            comments (Tuple[np.array[int], np.array[int], List[str]])

        Returns:
            Tuple[np.array[int], np.array[int], List[str]]
        """
        # TODO: get rid of the third element here and elsewhere?
        # extract content and comments
        if 'content' in self.to_extract and 'comments' in self.to_extract:
            if self.max_block_weight is None:
                return (np.logical_or(content[0], comments[0]).astype(int),
                        content[1],
                        content[2] + comments[2])
            else:
                return (np.logical_or(content[0], comments[0]).astype(int),
                        np.minimum(content[1], self.max_block_weight),
                        content[2] + comments[2])
        # extract content only
        elif 'content' in self.to_extract:
            if self.max_block_weight is None:
                return content
            else:
                return (content[0], np.minimum(content[1], self.max_block_weight), content[2])
        # extract comments only
        else:
            if self.max_block_weight is None:
                return comments
            else:
                return (comments[0], np.minimum(comments[1], self.max_block_weight), comments[2])

    def extract(self, html, encoding=None, as_blocks=False):
        """
        Extract the main content and/or comments from an HTML document and
        return it as a string or as a sequence of block objects.

        Args:
            html (str): HTML document as a string.
            encoding (str): Encoding of ``html``. If None (encoding unknown), the
                original encoding will be guessed from the HTML itself.
            as_blocks (bool): If False, return the main content as a combined
                string; if True, return the content-holding blocks as a list of
                block objects.

        Returns:
            str or List[Block]
        """
        blocks = self.blockifier.blockify(html, encoding=encoding)
        return self.extract_from_blocks(blocks, as_blocks=as_blocks)

    def extract_from_blocks(self, blocks, as_blocks=False):
        """
        Extract the main content and/or comments from a sequence of (all) blocks
        and return it as a string or as a sequence of block objects.

        Args:
            blocks (List[Block]): Blockify'd HTML document.
            as_blocks (bool): If False, return the main content as a combined
                string; if True, return the content-holding blocks as a list of
                block objects.

        Returns:
            str or List[Block]
        """
        if not self._has_enough_blocks(blocks):
            if as_blocks is False:
                return ''
            else:
                return []
        features_mat = self.features.transform(blocks)
        if self.prob_threshold is None:
            preds = self.model.predict(features_mat)
        else:
            self._positive_idx = (
                self._positive_idx or list(self.model.classes_).index(1))
            preds = (self.model.predict_proba(features_mat) > self.prob_threshold)[:, self._positive_idx]
        if as_blocks is False:
            return '\n'.join(blocks[ind].text for ind in np.flatnonzero(preds))
        else:
            return [blocks[ind] for ind in np.flatnonzero(preds)]

    def predict(self, blocks):
        """
        Predict class (content=1 or not-content=0) of each block in a sequence.

        Args:
            blocks (List[Block]): Blockify'd HTML document.

        Returns:
            ``np.ndarray``: 1D array of block-level, binary predictions for
                content (1) or not-content (0).
        """
        features_mat = self.features.transform(blocks)
        if self.prob_threshold is None:
            return self.model.predict(features_mat)
        else:
            self._positive_idx = (
                self._positive_idx or list(self.model.classes_).index(1))
            return (self.model.predict_proba(features_mat) > self.prob_threshold)[:, self._positive_idx].astype(int)

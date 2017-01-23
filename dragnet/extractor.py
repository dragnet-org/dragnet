import logging

import numpy as np
from sklearn.pipeline import make_union  # FeatureUnion

from .compat import string_


class Extractor(object):
    """
    Args:
        blockifier (:class:`Blockifier`)
        features (``Features`` or List[``Features``])
        model (:class:``sklearn.base.ClassifierMixin``)
        to_extract (str or Sequence[str]): 'content' or 'comments', or both via
            ['content', 'comments']
        prob_threshold (float): Minimum prediction probability of a block being
            "content" for it actually be classified as such.
        max_block_weight (int): When training,
    """

    def __init__(self, blockifier, features, model,
                 to_extract='content', prob_threshold=0.5, max_block_weight=200):
        self.blockifier = blockifier
        if isinstance(features, (list, tuple)):
            self.features = make_union(*features)
        else:
            self.features = features
        self.model = model
        if isinstance(to_extract, string_):
            self.to_extract = (to_extract,)
        else:
            self.to_extract = tuple(to_extract)
        self.prob_threshold = prob_threshold
        self.max_block_weight = max_block_weight
        self._positive_idx = None

    def fit(self, blocks, labels, weights=None):
        features_mat = self.features.fit_transform(blocks)
        self.model.fit(features_mat, labels, sample_weight=weights)
        return self

    def concatenate_data(self, data):
        """
        see :func:`data_processing.prepare_data`

        Args:
            data

        Returns:
            Tuple[List[Block], np.array(int), np.array(int)]
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
        if 'content' in self.to_extract and 'comments' in self.to_extract:
            return (np.logical_or(content[0], comments[0]).astype(int),
                    content[1] if self.max_block_weight is None else np.minimum(content[1], self.max_block_weight),
                    content[2] + comments[2])
        elif 'content' in self.to_extract:
            if self.max_block_weight is None:
                return content
            else:
                return (content[0], np.minimum(content[1], self.max_block_weight), content[2])
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

    analyze = extract

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
            return ''
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

    analyze_from_blocks = extract_from_blocks

    def predict(self, blocks):
        """
        Used for training...
        """
        features_mat = self.features.transform(blocks)
        if self.prob_threshold is None:
            return self.model.predict(features_mat)
        else:
            self._positive_idx = (
                self._positive_idx or list(self.model.classes_).index(1))
            return (self.model.predict_proba(features_mat) > self.prob_threshold)[:, self._positive_idx]

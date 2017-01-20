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
        to_extract (str or Sequence[str])
    """

    def __init__(self, blockifier, features, model, to_extract='content'):
        self.blockifier = blockifier
        if isinstance(features, list):
            self.features = make_union(*features)
        else:
            self.features = features
        self.model = model
        if isinstance(to_extract, string_):
            self.to_extract = (to_extract,)
        else:
            self.to_extract = tuple(to_extract)
        self._positive_idx = None

    def fit(self, blocks, labels, weights=None):
        features_mat = self.features.fit_transform(blocks)
        return self.model.fit(features_mat, labels, sample_weight=weights)

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
        # content, comments = the list tuple elements in data.training_data
        if 'content' in self.to_extract:
            return content
        elif 'comments' in self.to_extract:
            return comments
        else:
            return (np.logical_or(content[0], comments[0]).astype(int),
                    content[1],
                    content[2] + comments[2])

    def extract(self, html, prob_threshold=0.5):
        """
        Args:
            html (str)
            prob_threshold (float): must be in [0.0, 1.0] or None

        Returns:
            str
        """
        blocks = self.blockifier.blockify(html)
        if not self._has_enough_blocks(blocks):
            return ''
        features_mat = self.features.transform(blocks)
        if prob_threshold is None:
            preds = self.model.predict(features_mat)
        else:
            self._positive_idx = (
                self._positive_idx or list(self.model.classes_).index(1))
            preds = (self.model.predict_proba(features_mat) > prob_threshold)[:, self._positive_idx]
        return '\n'.join(blocks[ind].text for ind in np.flatnonzero(preds))

    analyze = extract

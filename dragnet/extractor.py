import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier

from .compat import string_, str_cast, unicode_
from .util import get_and_union_features
from .blocks import TagCountNoCSSReadabilityBlockifier


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

    def __init__(self, blockifier=TagCountNoCSSReadabilityBlockifier,
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

    def fit(self, documents, labels, weights=None):
        """
        Fit :class`Extractor` features and model to a training dataset.

        Args:
            blocks (List[Block])
            labels (``np.ndarray``)
            weights (``np.ndarray``)

        Returns:
            :class`Extractor`
        """
        block_groups = np.array([self.blockifier.blockify(doc) for doc in documents])
        mask = [self._has_enough_blocks(blocks) for blocks in block_groups]
        block_groups = block_groups[mask]
        labels = np.concatenate(np.array(labels)[mask])

        # TODO: This only 'fit's one doc at a time. No feature fitting actually
        # happens for now, but this might be important if the features change
        features_mat = np.concatenate([self.features.fit_transform(blocks)
                                       for blocks in block_groups])
        if weights is None:
            self.model.fit(features_mat, labels)
        else:
            weights = np.concatenate(np.array(weights)[mask])
            self.model.fit(features_mat, labels, sample_weight=weights)
        return self

    def get_html_labels_weights(self, data):
        """
        Gather the html, labels, and weights of many files' data.
        Primarily useful for training/testing an :class`Extractor`.

        Args:
            data: Output of :func:`dragnet.data_processing.prepare_all_data`.

        Returns:
            Tuple[List[Block], np.array(int), np.array(int)]: All blocks, all
                labels, and all weights, respectively.
        """
        all_html = []
        all_labels = []
        all_weights = []
        for html, content, comments in data:
            all_html.append(html)
            labels, weights = self._get_labels_and_weights(
                content, comments)
            all_labels.append(labels)
            all_weights.append(weights)
        return np.array(all_html), np.array(all_labels), np.array(all_weights)

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
        # extract content and comments
        if 'content' in self.to_extract and 'comments' in self.to_extract:
            labels = np.logical_or(content[0], comments[0]).astype(int)
            weights = content[1],
        # extract content only
        elif 'content' in self.to_extract:
            labels = content[0]
            weights = content[1]
        # extract comments only
        else:
            labels = comments[0]
            weights = comments[1]
        if self.max_block_weight is None:
            weights = np.minimum(weights, self.max_block_weight)

        return labels, weights

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
        preds, blocks = self.predict(html, encoding=encoding, return_blocks=True)
        if as_blocks is False:
            return str_cast(b'\n'.join(blocks[ind].text for ind in np.flatnonzero(preds)))
        else:
            return [blocks[ind] for ind in np.flatnonzero(preds)]


    def predict(self, documents, **kwargs):
        """
        Predict class (content=1 or not-content=0) of the blocks in one or many
        HTML document(s).

        Args:
            documents (str or List[str]): HTML document(s)

        Returns:
            ``np.ndarray`` or List[``np.ndarray``]: array of binary predictions
                for content (1) or not-content (0).
        """
        if isinstance(documents, (str, bytes, unicode_, np.unicode_)):
            return self._predict_one(documents, **kwargs)
        else:
            return np.concatenate([self._predict_one(doc, **kwargs) for doc in documents])


    def _predict_one(self, document, encoding=None, return_blocks=False):
        """
        Predict class (content=1 or not-content=0) of each block in an HTML
        document.

        Args:
            documents (str): HTML document

        Returns:
            ``np.ndarray``: array of binary predictions for content (1) or
            not-content (0).
        """
        # blockify
        blocks = self.blockifier.blockify(document, encoding=encoding)
        # get features
        try:
            features = self.features.transform(blocks)
        except ValueError: # Can't make features, predict no content
            preds = np.zeros((len(blocks)))
        # make predictions
        else:
            if self.prob_threshold is None:
                preds = self.model.predict(features)
            else:
                self._positive_idx = (
                    self._positive_idx or list(self.model.classes_).index(1))
                preds = self.model.predict_proba(features) > self.prob_threshold
                preds = preds[:, self._positive_idx].astype(int)

        if return_blocks:
            return preds, blocks
        else:
            return preds


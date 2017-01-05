#! /usr/bin/env python
"""
Implementations of the features interface.

A feature is a callable -- ``feature(List[Block], train=bool)`` -- that takes a
a list of blocks (created by :func:`Blockifier.blockify()`) and returns a numpy
array of computed features with shape (num blocks, num features).

The optional boolean param ``train`` is only used in an initial pre-processing
state for training.

To (optionally) allow the feature to set itself from some data, implement
``feature.init_parms(computed_features)`` *and* ``features.set_params(ret)``,
where ``computed_features`` is a call with ``train=True``, and ``ret`` is the
returned value from ``features.init_params``.
"""
import re
import numpy as np

from .compat import range_, string_


def normalize_features(features, mean_std):
    """Normalize the features IN PLACE.

       This a utility function.

       mean_std = {'mean':[list of means],
                   'std':[list of std] }
       the lists are the same length as features.shape[1]

       if features is None, then do nothing"""
    if features is not None:
        for k in range_(features.shape[1]):
            features[:, k] = (features[:, k] - mean_std['mean'][k]) / mean_std['std'][k]


class NormalizedFeature(object):
    """Normalize a feature with mean/std

    This is an abstraction of a normalized feature
    It acts sort of like a decorator on anything
    that implements the feature interface.

    Instances of this object also implement the feature interface"""

    def __init__(self, feature_to_normalize, mean_std=None):
        """feature_to_normalize = implements feature interface
        mean_std = a json blob of mean, std or a string with the file location
            or None"""
        self._feature = feature_to_normalize
        self._mean_std = NormalizedFeature.load_mean_std(mean_std)

    def __call__(self, blocks, train=False):
        # compute features and normalize
        if not train and self._mean_std is None:
            raise ValueError("You must provide mean_std or call init_params")

        features = self._feature(blocks)
        if not train:
            normalize_features(features, self._mean_std)

        return features

    def init_params(self, features):
        self._mean_std = {'mean': features.mean(axis=0),
                          'std': features.std(axis=0)}
        return self._mean_std

    def set_params(self, mean_std):
        # assert len(mean_std['mean']) == self.nfeatures
        # assert len(mean_std['std']) == self.nfeatures
        assert len(mean_std['std']) == len(mean_std['mean'])
        self._mean_std = mean_std

    @staticmethod
    def load_mean_std(mean_std):
        """mean_std is either a filename string,
        or a json blob.
        if a string, load it, otherwise just return"""
        import json
        if isinstance(mean_std, string_):
            return json.load(open(mean_std, 'r'))
        else:
            return mean_std


class CSSFeatures(object):
    """Class of features from id/class attributes.

    The features are 0/1 flags whether the attributes have
    a give set of tokens"""

    # we have set of tokens that we search for in each attribute.
    # The features are 0/1 flags whether these tokens appear in the CSS tags
    attribute_tokens = (
        ('id',
         ('nav', 'ss', 'top', 'content', 'link', 'title', 'comment', 'tools',
          'rating', 'ss')
         ),
        ('class',
         ('menu', 'widget', 'nav', 'share', 'facebook', 'cat', 'top', 'content',
          'item', 'twitter', 'button', 'title', 'header', 'ss', 'post',
          'comment', 'meta', 'alt', 'time', 'depth', 'thread', 'author', 'tools',
          # these two strings were implicitly concatenated
          # this is a bug, and it will be fixed
          'reply' + 'url',
          'avatar',
          # this is a duplicate entry (see above)
          'ss')
         )
        )

    def __call__(self, blocks, train=False):
        feature_vecs = (
            np.array(tuple(re.search(token, block.css[attrib]) is not None
                           for block in blocks))
            for attrib, tokens in self.attribute_tokens
            for token in tokens
            )
        return np.column_stack(tuple(feature_vecs))

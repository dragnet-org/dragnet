#! /usr/bin/env python

import re
import numpy as np
import scipy.weave
from .kohlschuetter import kohlschuetter_features

# implementations of the features interface.
#
# feature is a callable feature(list_of_blocks, train=False)
#   that takes list of blocks and returns a numpy array of computed_features
#       (len(blocks), nfeatures)
# The optional keyword "train" that is only called in an initial
#   pre-processing state for training
#
# It has an attribute "feature.nfeatures" that gives number of features
#
# To allow the feature to set itself from some data, feature can optionally
#   implement 
#       feature.init_parms(computed_features) AND
#       features.set_params(ret)
#   where computed_features is a call with train=True,
#   and ret is the returned value from features.init_params 
#   


def normalize_features(features, mean_std):
    """Normalize the features IN PLACE.

       This a utility function.

       mean_std = {'mean':[list of means],
                   'std':[list of std] }
        mean_std HAS AN OPTIONAL 'log' key.
            If present, it includes a flag whether to take a log first.
            If not None, then it gives a value to do a transform:
                log(x + exp(-value)) + value
       the lists are the same length as features.shape[1]
       
       if features is None, then do nothing"""
    if features is not None:
        if 'log' in mean_std:
            for k in xrange(features.shape[1]):
                if mean_std['log'][k] is not None:
                    features[:, k] = np.log(features[:, k] + np.exp(-mean_std['log'][k])) + mean_std['log'][k]

        for k in xrange(features.shape[1]):
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
        self.nfeatures = feature_to_normalize.nfeatures

    def __call__(self, blocks, train=False):
        # compute features and normalize
        if not train and self._mean_std is None:
            raise ValueError("You must provide mean_std or call init_params")

        features = self._feature(blocks)
        if not train:
            normalize_features(features, self._mean_std)

        return features

    def init_params(self, features):
        self._mean_std = {'mean':features.mean(axis=0),
                          'std':features.std(axis=0) }
        return self._mean_std

    def set_params(self, mean_std):
        assert len(mean_std['mean']) == self.nfeatures
        assert len(mean_std['std']) == self.nfeatures
        self._mean_std = mean_std

    @staticmethod
    def load_mean_std(mean_std):
        """mean_std is either a filename string,
        or a json blob.
        if a string, load it, otherwise just return"""
        import json
        if isinstance(mean_std, basestring):
            return json.load(open(mean_std, 'r'))
        else:
            return mean_std



re_capital = re.compile('[A-Z]')
re_digit = re.compile('\d')
def capital_digit_features(blocks, train=False):
    """percent of block that is capitalized and numeric"""
    features = np.zeros((len(blocks), 2))
    features[:, 0] = [len(re_capital.findall(ele.text)) / float(len(ele.text)) for ele in blocks]
    features[:, 1] = [len(re_digit.findall(ele.text)) / float(len(ele.text)) for ele in blocks]
    return features
capital_digit_features.nfeatures = 2


def token_feature(blocks, train=False):
    """A global token count feature"""
    from collections import defaultdict
    word_dict = defaultdict(lambda: 0)
    block_tokens = []
    token_count = 0
    for block in blocks:
        block_tokens.append(re.split('\W', block.text.strip()))
        for token in block_tokens[-1]:
            word_dict[token] += 1
            token_count += 1

    token_count = float(token_count)

    nblocks = len(blocks)
    feature = np.zeros((nblocks, 1))
    for k in xrange(nblocks):
        ntokens = len(block_tokens[k])

        for token in block_tokens[k]:
            feature[k] += np.log(word_dict[token])

        feature[k] = feature[k] / ntokens - np.log(token_count)

        if np.isinf(feature[k]):
            feature[k] = -10.0   # just in case

    return feature
token_feature.nfeatures = 1

class CSSFeatures(object):
    """Class of features from id/class attributes.

    The features are 0/1 flags whether the attributes have
    a give set of tokens"""

    # we have set of tokens that we search for in each
    # attribute.
    # The features are 0/1 flags whether these tokens
    # appear in the CSS tags
    attribute_tokens = {'id':['nav',
                              'ss',
                              'top',
                              'content',
                              'link',
                              'title',
                              'comment',
                              'tools',
                              'rating',
                              'ss'],
                     'class':['menu',
                              'widget',
                              'nav',
                              'share',
                              'facebook',
                              'cat',
                              'top',
                              'content',
                              'item',
                              'twitter',
                              'button',
                              'title',
                              'header',
                              'ss',
                              'post',
                              'comment',
                              'meta',
                              'alt',
                              'time',
                              'depth',
                              'thread',
                              'author',
                              'tools',
                              'reply'
                              'url',
                              'avatar',
                              'ss']}

    _attribute_order = ['id', 'class']

    nfeatures = sum(len(ele) for ele in attribute_tokens.itervalues())

    def __call__(self, blocks, train=False):
        ret = np.zeros((len(blocks), CSSFeatures.nfeatures))
        feature = 0
        for attrib in CSSFeatures._attribute_order:
            for token in CSSFeatures.attribute_tokens[attrib]:
                ret[:, feature] = [re.search(token, block.css[attrib]) is not None for block in blocks]
                feature += 1
        return ret



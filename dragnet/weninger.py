"""
inspired by
Weninger, Tim, William H. Hsu, and Jiawei Han. "CETR: content extraction via
    tag ratios." Proceedings of the 19th international conference on
    World wide web. ACM, 2010.
"""
from __future__ import absolute_import

import numpy as np

from .content_extraction_model import ContentExtractionModel
from .blocks import TagCountBlockifier
from .kmeans import KMeansFixedOrigin
from ._weninger import weninger_sx_sdx


def weninger_features(blocks, train=False):
    """Compute the content to tag ratio.
    Returns [smoothed ratio, absolute difference smooth ratio]
    """
    block_lengths = np.array(
        [len(block.text) for block in blocks], dtype=np.float)
    tagcounts = np.maximum(np.array(
        [block.features['tagcount'] for block in blocks]), 1.0)
    ctr = block_lengths / tagcounts
    return weninger_sx_sdx(ctr)


class WeningerKMeanModel(object):
    """Mock out the model interface for Weninger k-means"""
    def __init__(self, clusters=3):
        self._clusters = clusters

    def predict(self, features):
        assert features.shape[1] == 2
        km = KMeansFixedOrigin(self._clusters)
        km.fit(features)
        content = km.closest_centers(features) > 0
        return content.astype(np.int).reshape(-1, 1)


def weninger_features_kmeans(blocks, train=False):
    """The entire k-means prediction from Weninger as a feature"""
    w = WeningerKMeanModel(3)
    sx_sdx = weninger_features(blocks, train)
    return w.predict(sx_sdx)


class Weninger(ContentExtractionModel):

    def __init__(self, clusters=3, blockifier=TagCountBlockifier, **kwargs):
        features = [weninger_features]
        ContentExtractionModel.__init__(self, blockifier, features, WeningerKMeanModel(clusters), **kwargs)

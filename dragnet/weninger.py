
# inspired by
#Weninger, Tim, William H. Hsu, and Jiawei Han. "CETR: content extraction via
#    tag ratios." Proceedings of the 19th international conference on
#    World wide web. ACM, 2010.

import numpy as np
from scipy.ndimage import gaussian_filter
from .content_extraction_model import ContentExtractionModel
from .blocks import TagCountBlockifier
from .kmeans import KMeansFixedOrigin

def weninger_sx_sdx(x):
    """blocks = a 1D array of something (their content-tag-ratios, length, ..)
    Computes and returns the smoothed version of x and
    absolute smoothed discount (eqn 4)"""
    sigma = 1.0
    alpha = 3

    nx = len(x)

    # compute absolute discount
    dx = np.zeros(nx)
    for k in xrange(nx-1):
        dx[k] = abs(np.mean(x[(k + 1):min(k + 1 + alpha, nx)]) - x[k])
    dx[nx-1] = abs(np.mean(x[-2:]) - x[nx-1])

    # smooth
    sx = gaussian_filter(x, sigma=sigma)
    sdx = gaussian_filter(dx, sigma=sigma)

    return np.hstack((sx.reshape(-1, 1), sdx.reshape(-1, 1)))


def weninger_features(blocks, train=False):
    """Compute the content to tag ratio.
    Returns [smoothed ratio, absolute difference smooth ratio]
    """
    block_lengths = np.array([len(block.text) for block in blocks], dtype=np.float)
    tagcounts = np.maximum(np.array([block.features['tagcount'] for block in blocks]), 1.0)
    ctr = block_lengths / tagcounts
    sx_sdx = weninger_sx_sdx(ctr)
    #sx_sdx = np.minimum(sx_sdx, 300)  ###
    #sx_sdx = np.log(sx_sdx + 1.0)  ###
    return sx_sdx
weninger_features.nfeatures = 2

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
weninger_features_kmeans.nfeatures = 1

class Weninger(ContentExtractionModel):
    def __init__(self, clusters=3, blockifier=TagCountBlockifier, **kwargs):
        features = [weninger_features]
        ContentExtractionModel.__init__(self, blockifier, features, WeningerKMeanModel(clusters), **kwargs)


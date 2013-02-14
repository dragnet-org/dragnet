
# inspired by
#Weninger, Tim, William H. Hsu, and Jiawei Han. "CETR: content extraction via
#    tag ratios." Proceedings of the 19th international conference on
#    World wide web. ACM, 2010.

import numpy as np
from scipy.ndimage import gaussian_filter
from .content_extraction_model import ContentExtractionModel, IdentityPredictor
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
#weninger_sx_sdx.nfeatures = 2


class WeningerFeatures(object):
    nfeatures = 1

    def __init__(self, clusters=3):
        self._clusters = clusters

    def __call__(self, blocks, train=False):
        # make the content to tag ratio
        block_lengths = np.array([len(block.text) for block in blocks])
        tagcounts = np.array([block.features['tagcount'] for block in blocks])
        ctr = block_lengths / tagcounts
        sx_sdx = weninger_sx_sdx(ctr)

        km = KMeansFixedOrigin(self._clusters)
        km.fit(sx_sdx)
        content = km.closest_centers(sx_sdx) > 0
        return content.astype(np.int).reshape(-1, 1)


class Weninger(ContentExtractionModel):
    def __init__(self, clusters=3, blockifier=TagCountBlockifier, **kwargs):
        features = [WeningerFeatures(clusters)]
        ContentExtractionModel.__init__(self, blockifier, features, IdentityPredictor, **kwargs)




#! /usr/bin/env python

from .arias import AriasFeatures, Arias
from .blocks import Blockifier, PartialBlock, BlockifyError
from .features import NormalizedFeature, CSSFeatures
from .content_extraction_model import ContentExtractionModel
from .kohlschuetter import kohlschuetter_features, kohlschuetter
from .util import evaluation_metrics
from .weninger import weninger_features


class AllFeatures(object):
    """Easy access to feature instances.
    
    We need a way to get instances of the feature classes.
    Since these classes are potentially mutated by clients,
    we create a new instance on each access"""

    @staticmethod
    def get(key, *args, **kwargs):
        if key == 'kohlschuetter':
            return NormalizedFeature(kohlschuetter_features)
        elif key == 'css':
            return CSSFeatures()
        elif key == 'arias':
            return AriasFeatures(*args, **kwargs)
        elif key == 'weninger':
            return NormalizedFeature(weninger_features)
        else:
            raise KeyError


# we want to maintain backward compatibility with 
# code that uses the old interface to LogisticRegression
# and DragnetModel until existing
# code can be updated to use the new interface
from mozsci.models import LogisticRegression as lr
class LogisticRegression(lr):
    def pred(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


class DragnetModelKohlschuetterFeatures(ContentExtractionModel):
    def __init__(self, block_model, mean_std, threshold=0.5):
        """block_model = LogisticRegression
        mean_std = {'mean':[mean0, mean1, ..], 'std':[std0, std1, ..]}
        """
        koh_features = NormalizedFeature(kohlschuetter_features, mean_std)
        ContentExtractionModel.__init__(self, Blockifier,
                         [koh_features], block_model, threshold=threshold)


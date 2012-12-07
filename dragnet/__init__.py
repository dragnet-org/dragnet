#! /usr/bin/env python

from .arias import Arias
from .blocks import Blockifier, PartialBlock, BlockifyError
from .features import NormalizedFeature, all_features
from .content_extraction_model import ContentExtractionModel
from .kohlschuetter import kohlschuetter_features, kohlschuetter
from .util import evaluation_metrics


# we want to maintain backward compatibility with 
# code that the old interface to LogisticRegression
# and DragnetModeluses the old LogisticRegression and
# the existing model classses until existing
# code can be updated to use the new interface
#
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


from dragnet.arias import AriasFeatures, Arias
from dragnet.blocks import Blockifier, PartialBlock, BlockifyError
from dragnet.features import NormalizedFeature, CSSFeatures
from dragnet.content_extraction_model import ContentExtractionModel
from dragnet.kohlschuetter import kohlschuetter_features, kohlschuetter
from dragnet.util import evaluation_metrics
from dragnet.weninger import weninger_features_kmeans
from dragnet.readability import readability_features
from dragnet.models import content_extractor, content_comments_extractor


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
            return weninger_features_kmeans
        elif key == 'readability':
            return NormalizedFeature(readability_features)
        else:
            raise KeyError

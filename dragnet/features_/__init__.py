from dragnet.features_.base import StandardizedFeature
from dragnet.features_.css import CSSFeatures
from dragnet.features_.kohlschuetter import KohlschuetterFeatures
from dragnet.features_.readability import ReadabilityFeatures
from dragnet.features_.weninger import WeningerFeatures, ClusteredWeningerFeatures


def get_feature(name):
    """Get an instance of a ``Features`` class by ``name`` (str)."""
    if name == 'css':
        return CSSFeatures()
    elif name == 'kohlschuetter':
        return KohlschuetterFeatures()
    elif name == 'readability':
        return ReadabilityFeatures()
    elif name == 'weninger':
        return WeningerFeatures()
    elif name == 'clustered_weninger':
        return ClusteredWeningerFeatures()
    else:
        raise ValueError('invalid feature name: "{}"'.format(name))

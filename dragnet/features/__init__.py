from dragnet.features.standardized import StandardizedFeature
from dragnet.features.css import CSSFeatures
from dragnet.features.kohlschuetter import KohlschuetterFeatures
from dragnet.features.readability import ReadabilityFeatures
from dragnet.features.weninger import WeningerFeatures, ClusteredWeningerFeatures


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

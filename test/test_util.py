# import unittest

from sklearn.pipeline import FeatureUnion

from dragnet.features import KohlschuetterFeatures, WeningerFeatures
from dragnet import util


def test_evaluation_metrics():
    predicted = 'skiing sparkling soft snow in soft sun'.split()
    actual = 'soft snow in soft sun soft turns turns'.split()

    def _f1(p, r):
        return 2 * p * r / (p + r)

    # for bag of words assumption
    p = 4.0 / 6.0
    r = 4.0 / 5
    f1 = _f1(p, r)

    prf = util.evaluation_metrics(predicted, actual)
    assert (p, r, f1) == prf

    # for list assumption
    p = 5 / 7.0
    r = 5 / 8.0
    f1 = _f1(p, r)
    prf = util.evaluation_metrics(predicted, actual, bow=False)
    assert (p, r, f1) == prf


def test_get_and_union_features_str():
    features = util.get_and_union_features('weninger')
    assert isinstance(features, WeningerFeatures)


def test_get_and_union_features_strs():
    features = util.get_and_union_features(['weninger', 'kohlschuetter'])
    assert isinstance(features, FeatureUnion)
    assert [t[0] for t in features.transformer_list] == ['weninger', 'kohlschuetter']


def test_get_and_union_features_instance():
    features = util.get_and_union_features(WeningerFeatures())
    assert isinstance(features, WeningerFeatures)


def test_get_and_union_features_instances():
    features = util.get_and_union_features([WeningerFeatures(), KohlschuetterFeatures()])
    assert isinstance(features, FeatureUnion)
    assert [t[0] for t in features.transformer_list] == ['weningerfeatures', 'kohlschuetterfeatures']


def test_get_and_union_features_tuple():
    features = util.get_and_union_features(
        [('feat1', WeningerFeatures()),
         ('feat2', KohlschuetterFeatures())]
    )
    assert isinstance(features, FeatureUnion)
    assert [t[0] for t in features.transformer_list] == ['feat1', 'feat2']

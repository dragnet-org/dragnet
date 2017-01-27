import unittest

from sklearn.pipeline import FeatureUnion

from dragnet.features import KohlschuetterFeatures, WeningerFeatures
from dragnet import util


class UtilTestCase(unittest.TestCase):

    def test_evaluation_metrics(self):

        predicted = 'skiing sparkling soft snow in soft sun'.split()
        actual = 'soft snow in soft sun soft turns turns'.split()

        def _f1(p, r):
            return 2 * p * r / (p + r)

        # for bag of words assumption
        p = 4.0 / 6.0
        r = 4.0 / 5
        f1 = _f1(p, r)

        prf = util.evaluation_metrics(predicted, actual)
        self.assertEqual((p, r, f1), prf)

        # for list assumption
        p = 5 / 7.0
        r = 5 / 8.0
        f1 = _f1(p, r)
        prf = util.evaluation_metrics(predicted, actual, bow=False)
        self.assertEqual((p, r, f1), prf)

    def test_get_and_union_features_str(self):
        features = util.get_and_union_features('weninger')
        self.assertIsInstance(features, WeningerFeatures)

    def test_get_and_union_features_strs(self):
        features = util.get_and_union_features(['weninger', 'kohlschuetter'])
        self.assertIsInstance(features, FeatureUnion)
        self.assertEqual(
            [t[0] for t in features.transformer_list],
            ['weninger', 'kohlschuetter'])

    def test_get_and_union_features_instance(self):
        features = util.get_and_union_features(WeningerFeatures())
        self.assertIsInstance(features, WeningerFeatures)

    def test_get_and_union_features_instances(self):
        features = util.get_and_union_features([WeningerFeatures(), KohlschuetterFeatures()])
        self.assertIsInstance(features, FeatureUnion)
        self.assertEqual(
            [t[0] for t in features.transformer_list],
            ['weningerfeatures', 'kohlschuetterfeatures'])

    def test_get_and_union_features_tuple(self):
        features = util.get_and_union_features(
            [('feat1', WeningerFeatures()),
             ('feat2', KohlschuetterFeatures())]
            )
        self.assertIsInstance(features, FeatureUnion)
        self.assertEqual(
            [t[0] for t in features.transformer_list],
            ['feat1', 'feat2'])


if __name__ == "__main__":
    unittest.main()

import io
import json
import os
import unittest

import numpy as np

from dragnet.blocks import TagCountReadabilityBlockifier
from dragnet.features import _readability

FIXTURES = os.path.join('test', 'datafiles')


class TestReadabilityFeatures(unittest.TestCase):

    def test_readability_make_readability_features(self):
        with io.open(os.path.join(FIXTURES, 'models_testing.html')) as fin:
            html = fin.read()
        blks = TagCountReadabilityBlockifier.blockify(html)
        actual_features = _readability.make_readability_features(blks)
        with io.open(os.path.join(FIXTURES, 'readability_features.json')) as fin:
            expected_features = np.array(json.loads(fin.read()))
        self.assertTrue(
            np.allclose(actual_features.flatten(), expected_features, rtol=0.0005))
        self.assertEqual(actual_features.shape[1], 1)


if __name__ == '__main__':
    unittest.main()

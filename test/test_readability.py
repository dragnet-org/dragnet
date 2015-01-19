
import unittest
import os
import json

import numpy as np

from dragnet.blocks import TagCountReadabilityBlockifier
from dragnet import readability

FIXTURES = 'test/datafiles'

class TestReadabilityFeatures(unittest.TestCase):
    def test_readability(self):
        with open(os.path.join(FIXTURES, 'models_testing.html')) as fin:
            html = fin.read()
        blks = TagCountReadabilityBlockifier.blockify(html)
        actual_features = readability.readability_features(blks)

        with open(os.path.join(FIXTURES, 'readability_features.json')) as fin:
            expected_features = np.array(json.loads(fin.read()))

        self.assertTrue(
            np.allclose(actual_features.flatten(), expected_features))

if __name__ == '__main__':
    unittest.main()


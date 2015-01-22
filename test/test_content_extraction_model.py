
import unittest
from dragnet import Blockifier, kohlschuetter, ContentExtractionModel, NormalizedFeature, kohlschuetter_features
import re
import numpy as np
from mozsci.models import LogisticRegression
from html_for_testing import big_html_doc

class TestContentExtractionModel(unittest.TestCase):
    def test_dragnet_model(self):
        params = {'b':0.2, 'w':[0.4, -0.2, 0.9, 0.8, -0.3, -0.5]}
        block_model = LogisticRegression.load_model(params)
        mean_std = {'mean':[0.0, 0.1, 0.2, 0.5, 0.0, 0.3], 'std':[1.0, 2.0, 0.5, 1.2, 0.75, 1.3]}
        koh_features = NormalizedFeature(kohlschuetter_features, mean_std)

        dm = ContentExtractionModel(Blockifier, [koh_features], block_model, threshold=0.5)
        content = dm.analyze(big_html_doc)

        # make prediction from individual components
        # to do so, we use kohlschuetter.make_features and LogisticRegression
        features, blocks = kohlschuetter.make_features(big_html_doc)
        nblocks = len(blocks)
        features_normalized = np.zeros(features.shape)
        for k in xrange(6):
            features_normalized[:, k] = (features[:, k] - mean_std['mean'][k]) / mean_std['std'][k]
        blocks_keep_indices = np.arange(nblocks)[block_model.predict(features_normalized) > 0.5]

        actual_content = ' '.join([blocks[index].text for index in blocks_keep_indices])

        # check that the tokens are the same!
        self.assertEqual(re.split('\s+', actual_content.strip()),
                        re.split('\s+', content.strip()))


if __name__ == "__main__":
    unittest.main()


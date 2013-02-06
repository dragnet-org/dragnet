
import unittest
import numpy as np
from scipy import percentile

from html_for_testing import big_html_doc
from dragnet import Blockifier, Arias


class TestArias(unittest.TestCase):
    def test_arias_model(self):
        cutoff_percent = 60
        window = 2

        a = Arias(cutoff_percent, window)
        content_arias = a.analyze(big_html_doc)

        # now compute the actual content
        blocks = Blockifier.blockify(big_html_doc)
        actual_content_indices = [1, 2, 3]
        actual_content = ' '.join([blocks[k].text for k in actual_content_indices])

        self.assertEqual(actual_content, content_arias)


if __name__ == "__main__":
    unittest.main()





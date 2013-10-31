
import unittest
import json
import os.path
from dragnet.models import * 

FIXTURES = 'test/datafiles'

class TestModels(unittest.TestCase):
    def test_models(self):

        html_string = open(os.path.join(
            FIXTURES, 'models_testing.html'), 'r').read()

        models = [kohlschuetter_model,
                  weninger_model, 
                  kohlschuetter_weninger_model,
                  kohlschuetter_css_model,
                  kohlschuetter_css_weninger_model]

        content = [m.analyze(html_string) for m in models]

        actual_content = json.load(open(
            os.path.join(FIXTURES, 'models_content.json'), 'r'))

        for k in xrange(len(content)):
            self.assertTrue(actual_content[k] == content[k])

if __name__ == "__main__":
    unittest.main()



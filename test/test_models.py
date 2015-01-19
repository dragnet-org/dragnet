
import unittest
import json
import os.path
from dragnet.models import * 

FIXTURES = 'test/datafiles'

class TestModels(unittest.TestCase):
    def test_models(self):

        html = open(os.path.join(
            FIXTURES, 'models_testing.html'), 'r').read()

        models = [kohlschuetter_model,
                  weninger_model, 
                  kohlschuetter_weninger_model,
                  kohlschuetter_css_model,
                  kohlschuetter_css_weninger_model,
                  content_extractor,
                  content_comments_extractor]

        actual_content = json.load(open(
            os.path.join(FIXTURES, 'models_content.json'), 'r'))

        for k in xrange(len(models)):
            # some of the models (weninger) aren't deterministic
            # so the content doesn't match exactly every time,
            # although it passes most of the time
            # we allow a max of 5 failures before failing the entire test
            m = models[k]
            passed = False
            for i in xrange(5):
                content = m.analyze(html)
                if actual_content[k].encode('utf-8') == content:
                    passed = True
                    break
            self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()



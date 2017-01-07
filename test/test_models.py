
import unittest
import json
import os.path

from dragnet.models import *
from dragnet.compat import range_

FIXTURES = 'test/datafiles'


class TestModels(unittest.TestCase):

    def setUp(self):
        self._html = open(os.path.join(
            FIXTURES, 'models_testing.html'), 'r').read()

    def test_models(self):
        # models = [kohlschuetter_model,
        #           weninger_model,
        #           kohlschuetter_weninger_model,
        #           kohlschuetter_css_model,
        #           kohlschuetter_css_weninger_model,
        #           content_extractor,
        #           content_comments_extractor]
        models = [content_extractor,
                  content_comments_extractor]

        actual_content = json.load(open(
            os.path.join(FIXTURES, 'models_content_mod.json'), 'r'))

        for k in range_(len(models)):
            # some of the models (weninger) aren't deterministic
            # so the content doesn't match exactly every time,
            # although it passes most of the time
            # we allow a max of 5 failures before failing the entire test
            m = models[k]
            passed = False
            for i in range_(10):
                content = m.analyze(self._html)
                if actual_content[k].encode('utf-8') == content:
                    passed = True
                    break
            self.assertTrue(passed)

    def test_content_and_content_comments_extractor(self):
        content = content_extractor.analyze(self._html)
        content_comments = content_comments_extractor.analyze(self._html)

        passed_content = False
        passed_content_comments = False
        for i in range_(10):
            actual_content, actual_content_comments = \
                content_and_content_comments_extractor.analyze(self._html)
            passed_content = actual_content == content
            passed_content_comments = (
                actual_content_comments == content_comments)
            if passed_content and passed_content_comments:
                break

        self.assertTrue(passed_content)
        self.assertTrue(passed_content_comments)

    def test_content_and_content_comments_extractor_blocks(self):
        '''
        The content and content/comments extractor should return proper blocks
        '''
        content = content_extractor.analyze(self._html, blocks=True)
        content_comments = content_comments_extractor.analyze(
            self._html, blocks=True)

        passed_content = False
        passed_content_comments = False
        for i in range_(5):
            actual_content, actual_content_comments = \
                content_and_content_comments_extractor.analyze(
                    self._html, blocks=True)
            passed_content = (
                [blk.text for blk in actual_content] ==
                [blk.text for blk in content]
                )
            passed_content_comments = (
                [blk.text for blk in actual_content_comments] ==
                [blk.text for blk in content_comments]
                )
            if passed_content and passed_content_comments:
                break

        self.assertTrue(passed_content)
        self.assertTrue(passed_content_comments)

if __name__ == "__main__":
    unittest.main()

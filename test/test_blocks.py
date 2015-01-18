
import unittest
from dragnet import blocks
from html_for_testing import big_html_doc

class Testencoding(unittest.TestCase):
    def test_guess_encoding(self):
        s = """<?xml version="1.0" encoding="ISO-8859-1"?>
        <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
          "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">

          <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="fr" lang="fr">
          """
        self.assertEqual(blocks.guess_encoding(s), 'ISO-8859-1')

        s = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN"
          "http://www.w3.org/TR/html4/strict.dtd">
                 
          <head>
          <meta http-equiv="content-type" content="text/html; charset=GB2312">
          </head>"""
        self.assertEqual(blocks.guess_encoding(s), 'GB2312')

        s = """<html>sadfsa</html>"""
        self.assertEqual(blocks.guess_encoding(s, 'asciI'), 'asciI')


class Test_text_subtree(unittest.TestCase):
    def test_text_from_subtree(self):
        from lxml import etree
        s = """<a href=".">WILL <img src="."> THIS PASS <b>THE TEST</b> ??</a>"""
        tree = etree.fromstring(s, etree.HTMLParser(recover=True))
        text_list = blocks.text_from_subtree(tree)
        text_str = ' '.join([ele.strip() for ele in text_list if ele.strip() != ''])
        self.assertEqual(text_str,
            'WILL THIS PASS THE TEST ??')

    def test_text_from_subtree_decode_error(self):
        from lxml import etree
        # this is an invalid utf-8 character
        s = '<div>\x92</div>'
        tree = etree.fromstring(s, etree.HTMLParser(recover=True, encoding='utf-8'))
        text_list = blocks.text_from_subtree(tree)
        text_str = ' '.join([ele.strip() for ele in text_list if ele.strip() != ''])
        self.assertEqual(text_str, '')


class Test_TagCountPB(unittest.TestCase):

    def check_tagcount(self, expected, predicted):
        self.assertEqual(predicted['tagcount'],
                        expected[0])
        self.assertEqual(predicted['tagcount_since_last_block'],
                expected[1])
        self.assertEqual(predicted['anchor_count'],
                expected[2])
        self.assertEqual(predicted['min_depth_since_last_block'],
                expected[3])

    def test_simple(self):
        s = """<html><body><div>some text <i>in italic</i> and something else
                    <script> <div>skip this</div> </script>
                    <b>bold stuff</b> after the script
               </div></body></html>"""
        blks = blocks.TagCountBlockifier.blockify(s)
        self.check_tagcount((3, 2, 0, 0), blks[0].features)
        self.assertTrue(len(blks) == 1)
        
    def test_big_html(self):
        blks = blocks.TagCountBlockifier.blockify(big_html_doc)

        actual_features = [
            (1, 2, 0, 0),
            (2, 0, 0, 2),
            (2, 0, 1, 3),
            (2, 0, 0, 3),  # blockquote
            (1, 2, 0, 3),
            (1, 0, 0, 3),
            (1, 0, 0, 3),
            (1, 2, 0, 2), # first comment
            (2, 0, 1, 4),
            (1, 1, 0, 3),
#            (3, 0, 1, 0)  # NOTE: this is a bug here.  It's due
                    # to the _tc-1 assumption in the feature extractor
                    # that fails for the last block. (we don't call
                    # tag_tagcount again before appending the block)
            ]

        for a, b in zip(actual_features, blks):
            self.check_tagcount(a, b.features)


class TestReadabilityBlocks(unittest.TestCase):
    def setUp(self):
        self.html_string = '''
            <html><body>
            <div class='content'>1 <i>i</i>
                <p class='meta'>2</p>
                <p>3</p>
                <div id='contact'>4
                    <p>5</p>
                    <p>6</p>
                </div>
                <div></div>
            </div>
            <h1>7</h1>
            </body></html>
            '''
    
    def test_ancestors(self):
        blks = blocks.TagCountReadabilityBlockifier.blockify(self.html_string)
        # get the text and ancestors from the blocks
        actual = [(blk.text, blk.features['ancestors']) for blk in blks]
        expected = [('1 i', [0, 2]),
            ('2', [0, 2, 4]), ('3', [0, 2, 4]), ('4', [0, 2, 4]),
            ('5', [0, 2, 4, 9]), ('6', [0, 2, 4, 9]),
            ('7', [0, 2])]
        self.assertEqual(actual, expected)

    def test_class_weights(self):
        blks = blocks.TagCountReadabilityBlockifier.blockify(self.html_string)
        actual = [blk.features['readability_class_weights'] for blk in blks]
        expected = [
            [(0, 0), (2, 0), (4, 30), (6, 0)], [(7, -25)], [(8, 0)],
            [(9, -20)], [(11, 0)], [(12, 0)], [(13, 5), (14, -5)]
        ]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()


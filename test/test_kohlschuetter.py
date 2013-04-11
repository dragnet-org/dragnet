
import unittest
from dragnet import Blockifier, PartialBlock, BlockifyError, kohlschuetter
from lxml import etree
import re
import numpy as np
from mozsci.models import LogisticRegression
from html_for_testing import big_html_doc

class KohlschuetterUnitBase(unittest.TestCase):
    def block_output_tokens(self, blocks, true_tokens):
        """blocks = the output from blockify
           true_tokens = a list of true tokens"""

        self.assertTrue(len(blocks) == len(true_tokens))

        for k in xrange(len(blocks)):
            block_tokens = re.split('\s+', blocks[k].text.strip())
            self.assertEqual(block_tokens, true_tokens[k])

    def link_output_tokens(self, blocks, true_tokens):
        self.assertTrue(len(blocks) == len(true_tokens))

        link_tokens = [ele.link_tokens for ele in blocks]
        for k in xrange(len(link_tokens)):
            self.assertEqual(link_tokens[k], true_tokens[k])

    def css_output_tokens(self, blocks, attrib, true_tokens):
        self.assertEqual(len(blocks), len(true_tokens))
        for k in xrange(len(blocks)):
            css_tokens = re.split('\s+', blocks[k].css[attrib].strip())
            self.assertEqual(css_tokens, true_tokens[k])


class TestBlockifier(KohlschuetterUnitBase):
    def test_lxml_error(self):
        """tests the case where lxml raises an error during parsing

        also handles case where lxml returns None for the tree"""
        # this raises an error in parsing
        self.assertRaises(etree.XMLSyntaxError, etree.fromstring, '', etree.HTMLParser(recover=True))
        self.assertRaises(BlockifyError, Blockifier.blockify, '')

        # this returns None in lxml
        self.assertTrue(None == etree.fromstring('<!--', etree.HTMLParser(recover=True)) )
        self.assertRaises(BlockifyError, Blockifier.blockify, '<!--')

    def test_very_simple(self):
        """test_very_simple"""
        s = """<div>some text
                    <script> skip this </script>
                    more text here
               </div>"""
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(blocks, [['some', 'text', 'more', 'text', 'here']])

    def test_very_simple2(self):
        s = """<div>some text <i>in italic</i> and something else
                    <script> <div>skip this</div> </script>
                    <b>bold stuff</b> after the script
               </div>"""
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(blocks, [['some', 'text', 'in', 'italic', 'and', 'something', 'else', 'bold', 'stuff', 'after', 'the', 'script']])

    def test_simple_two_blocks(self):
        s = """<h1>A title <i>with italics</i> and other words</h1>
               some text outside the h1
               <div>a div <span class="test"> with a span </span> more </div>"""
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(blocks,
               [['A', 'title', 'with', 'italics', 'and', 'other', 'words', 'some', 'text', 'outside', 'the', 'h1'],
                ['a', 'div', 'with', 'a', 'span', 'more']])

    def test_comment(self):
        s = """<H1>h1 tag word</H1>
               <!-- a comment -->
               orphaned text
               <TABLE><tr><td>table data</td></tr><tr><td>second row</td></tr></TABLE>
               final
               """
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(blocks,
                [['h1', 'tag', 'word', 'orphaned', 'text'],
                 ['table', 'data', 'second', 'row', 'final']])

    def test_empty_blocks(self):
        s = """<div> .! </div>
                some text
               <h1> in an h1 </h1>
               <p> ! _ </p>
            """
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(blocks,
                    [['.!', 'some', 'text'], ['in', 'an', 'h1']])

    def test_nested_blocks(self):
        s = """initial text
            <div>div <p> with paragraph </p>
            after Paragraph
            <div> nested div <div> and again </div>here</div>
            </div>
            final
            <div> <i> italic </i> before <h1>tag</h1></div>"""
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(blocks,
                [['initial', 'text'],
                ['div'],
                ['with', 'paragraph', 'after', 'Paragraph'],
                ['nested', 'div'],
                ['and', 'again', 'here', 'final'],
                ['italic', 'before'],
                ['tag']])

    def test_anchors(self):
        s = """<a href=".">anchor text</a>
               more
               <div>text <a href=".">123</a><div>MORE!</div></div>
               an img link<a href="."><img src="."></a>there
               <table><tr><td><a href=".">WILL <img src="."> THIS PASS <b>THE TEST</b> ??</a></tr></td></table>"""
        blocks = Blockifier.blockify(s)

        self.block_output_tokens(blocks,
              [['anchor', 'text', 'more'],
              ['text', '123'],
              ['MORE!', 'an', 'img', 'link', 'there'],
              ['WILL', 'THIS', 'PASS', 'THE', 'TEST', '??']])

        self.link_output_tokens(blocks,
            [['anchor', 'text'],
             ['123'],
             [],
             ['WILL', 'THIS', 'PASS', 'THE', 'TEST', '??']])


    def test_unicode(self):
        s = u"""<div><div><a href="."> the registered trademark \xae</a></div></div"""
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(blocks,
            [['the', 'registered', 'trademark', u'\xae']])
        self.link_output_tokens(blocks,
            [['the', 'registered', 'trademark', u'\xae']])

    def test_all_non_english(self):
        s = u"""<div> <div> \u03b4\u03bf\u03b3 </div> <div> <a href="summer">\xe9t\xe9</a> </div>
         <div> \u62a5\u9053\u4e00\u51fa </div> </div>"""
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(blocks,
            [[u'\u03b4\u03bf\u03b3'],
            [u'\xe9t\xe9'],
            [u'\u62a5\u9053\u4e00\u51fa']])
        self.link_output_tokens(blocks,
            [[],
             [u'\xe9t\xe9'],
             []])

    def test_class_id(self):
        s = """<div CLASS='d1'>text in div
                <h1 id="HEADER">header</h1>
                <div class="nested">dragnet</div>
                </div>"""
        blocks = Blockifier.blockify(s)

        self.block_output_tokens(blocks,
            [['text', 'in', 'div'],
            ['header'],
            ['dragnet']])

        self.css_output_tokens(blocks, 'id',
            [[''],
             ['header'],
             ['']])

        self.css_output_tokens(blocks, 'class',
            [['d1'],
             [''],
             ['nested']])



    def test_big_html(self):
        s = big_html_doc
        blocks = Blockifier.blockify(s)

        self.block_output_tokens(blocks,
        [['Inside', 'the', 'h1', 'tag'],
        ['First', 'line', 'of', 'the', 'content', 'in', 'bold'],
        ['A', 'paragraph', 'with', 'a', 'link', 'and', 'some', 'additional', 'words.'],
        ['Second', 'paragraph', 'Insert', 'a', 'block', 'quote', 'here'],
        ['Some', 'more', 'text', 'after', 'the', 'image'],
        ['An', 'h2', 'tag', 'just', 'for', 'kicks'],
        ['Finally', 'more', 'text', 'at', 'the', 'end', 'of', 'the', 'content'],
        ['This', 'is', 'a', 'comment'],
        ['with', 'two', 'paragraphs', 'and', 'some', 'comment', 'spam'],
        ['Second', 'comment'],
        ['Footer', 'text']])

        self.link_output_tokens(blocks,
            [[],
            [],
            ['a', 'link'],
            [],
            [],
            [],
            [],
            [],
            ['and', 'some', 'comment', 'spam'],
            [],
            []])

        self.css_output_tokens(blocks, 'class',
            [[''],
            ['title'],
            ['link'],
            [''],
            [''],
            [''],
            [''],
            [''],
            [''],
            [''],
            ['footer']])


        self.css_output_tokens(blocks, 'id',
            [[''],
            ['content'],
            ['para'],
            [''],
            [''],
            [''],
            [''],
            [''],
            [''],
            [''],
            ['']])



class TestKohlschuetter(KohlschuetterUnitBase):
    def test_small_doc(self):
        self.assertEqual((None, []), kohlschuetter.make_features('<html></html>'))
        self.assertEqual('', kohlschuetter.analyze('<html></html>'))

        s = '<html> <p>a</p> <div>b</div> </html>'
        features, blocks = kohlschuetter.make_features(s)
        self.assertTrue(features is None)
        self.block_output_tokens(blocks, [['a'], ['b']])
        self.assertEqual('a b', kohlschuetter.analyze(s))


    def test_make_features(self):
        s = '<html> <p>first </p> <div> <p>second block with <a href=''>anchor</a> </p> <p>the third block</p> </div> </html>'
        features, blocks = kohlschuetter.make_features(s)
        self.block_output_tokens(blocks, [['first'], ['second', 'block', 'with', 'anchor'], ['the', 'third', 'block']])
        self.link_output_tokens(blocks, [[], ['anchor'], []])

        text_density = [1.0, 4.0, 3.0]
        link_density = [1.0, 0.25, 1.0 / 3.0]

        self.assertTrue(np.allclose(features[0, :], [0.0, 0.0, link_density[0], text_density[0], link_density[1], text_density[1]]))
        self.assertTrue(np.allclose(features[1, :], [link_density[0], text_density[0], link_density[1], text_density[1], link_density[2], text_density[2]]))
        self.assertTrue(np.allclose(features[2, :], [link_density[1], text_density[1], link_density[2], text_density[2], 0.0, 0.0]))



if __name__ == "__main__":
    unittest.main()



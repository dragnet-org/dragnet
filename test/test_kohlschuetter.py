import io
from lxml import etree
import os
import re
import unittest

import numpy as np

from dragnet import Blockifier, BlockifyError
from dragnet.features import KohlschuetterFeatures
from dragnet.compat import range_


with io.open(os.path.join('test', 'datafiles', 'HTML', 'page_for_testing.html'), 'r') as f:
    big_html_doc = f.read()


class KohlschuetterUnitBase(unittest.TestCase):

    def block_output_tokens(self, blocks, true_tokens):
        """blocks = the output from blockify
           true_tokens = a list of true tokens"""

        self.assertTrue(len(blocks) == len(true_tokens))

        for k in range_(len(blocks)):
            block_tokens = re.split('\s+', blocks[k].text.strip())
            self.assertEqual(block_tokens, true_tokens[k])

    def link_output_tokens(self, blocks, true_tokens):
        self.assertTrue(len(blocks) == len(true_tokens))

        link_tokens = [ele.link_tokens for ele in blocks]
        for k in range_(len(link_tokens)):
            self.assertEqual(link_tokens[k], true_tokens[k])

    def css_output_tokens(self, blocks, attrib, true_tokens):
        self.assertEqual(len(blocks), len(true_tokens))
        for k in range_(len(blocks)):
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
        self.assertTrue(etree.fromstring('<!--', etree.HTMLParser(recover=True)) is None)
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

    @staticmethod
    def count_divs(tree):
        div_xpath = etree.XPath("//div")
        TestBlockifier.div_count = len(div_xpath(tree))

    def test_callback(self):
        s = """<div>some text <i>in italic</i> and something else
                    <pre> <div>skip this</div> </pre>
                    <b>bold stuff</b> after the script
               </div>"""
        blocks = Blockifier.blockify(
            s, parse_callback=TestBlockifier.count_divs)
        self.assertEqual(TestBlockifier.div_count, 2)

    def test_simple_two_blocks(self):
        s = """<h1>A title <i>with italics</i> and other words</h1>
               some text outside the h1
               <div>a div <span class="test"> with a span </span> more </div>"""
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(
            blocks,
            [['A', 'title', 'with', 'italics', 'and', 'other', 'words', 'some', 'text', 'outside', 'the', 'h1'],
             ['a', 'div', 'with', 'a', 'span', 'more']]
            )

    def test_comment(self):
        s = """<H1>h1 tag word</H1>
               <!-- a comment -->
               orphaned text
               <TABLE><tr><td>table data</td></tr><tr><td>second row</td></tr></TABLE>
               final
               """
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(
            blocks,
            [['h1', 'tag', 'word', 'orphaned', 'text'],
             ['table', 'data', 'second', 'row', 'final']]
            )

    def test_empty_blocks(self):
        s = """<div> .! </div>
                some text
               <h1> in an h1 </h1>
               <p> ! _ </p>
            """
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(
            blocks, [['.!', 'some', 'text'], ['in', 'an', 'h1']])

    def test_nested_blocks(self):
        s = """initial text
            <div>div <p> with paragraph </p>
            after Paragraph
            <div> nested div <div> and again </div>here</div>
            </div>
            final
            <div> <i> italic </i> before <h1>tag</h1></div>"""
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(
            blocks,
            [['initial', 'text'],
             ['div'],
             ['with', 'paragraph', 'after', 'Paragraph'],
             ['nested', 'div'],
             ['and', 'again', 'here', 'final'],
             ['italic', 'before'],
             ['tag']]
            )

    def test_anchors(self):
        s = """<a href=".">anchor text</a>
               more
               <div>text <a href=".">123</a><div>MORE!</div></div>
               an img link<a href="."><img src="."></a>there
               <table><tr><td><a href=".">WILL <img src="."> THIS PASS <b>THE TEST</b> ??</a></tr></td></table>"""
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(
            blocks,
            [['anchor', 'text', 'more'],
             ['text', '123'],
             ['MORE!', 'an', 'img', 'link', 'there'],
             ['WILL', 'THIS', 'PASS', 'THE', 'TEST', '??']]
            )
        self.link_output_tokens(
            blocks,
            [['anchor', 'text'],
             ['123'],
             [],
             ['WILL', 'THIS', 'PASS', 'THE', 'TEST', '??']]
            )

    def test_unicode(self):
        s = u"""<div><div><a href="."> the registered trademark \xae</a></div></div>"""
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(
            blocks,
            [['the', 'registered', 'trademark', u'\xae'.encode('utf-8')]])
        self.link_output_tokens(
            blocks,
            [['the', 'registered', 'trademark', u'\xae'.encode('utf-8')]])

    def test_all_non_english(self):
        s = u"""<div> <div> \u03b4\u03bf\u03b3 </div> <div> <a href="summer">\xe9t\xe9</a> </div>
         <div> \u62a5\u9053\u4e00\u51fa </div> </div>"""
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(
            blocks,
            [[u'\u03b4\u03bf\u03b3'.encode('utf-8')],
             [u'\xe9t\xe9'.encode('utf-8')],
             [u'\u62a5\u9053\u4e00\u51fa'.encode('utf-8')]]
            )
        self.link_output_tokens(
            blocks,
            [[],
             [u'\xe9t\xe9'.encode('utf-8')],
             []]
            )

    def test_class_id(self):
        s = """<div CLASS='d1'>text in div
                <h1 id="HEADER">header</h1>
                <div class="nested">dragnet</div>
                </div>"""
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(
            blocks, [['text', 'in', 'div'], ['header'], ['dragnet']])
        self.css_output_tokens(
            blocks, 'id', [[''], ['header'], ['']])
        self.css_output_tokens(
            blocks, 'class', [['d1'], [''], ['nested']])

    def test_class_id_unicode(self):
        s = """<div CLASS=' class1 \xc2\xae'>text in div
                <h1 id="HEADER">header</h1>
                </div>"""
        blocks = Blockifier.blockify(s, encoding='utf-8')
        self.block_output_tokens(
            blocks, [['text', 'in', 'div'], ['header']])
        self.css_output_tokens(
            blocks, 'id', [[''], ['header']])
        self.css_output_tokens(
            blocks, 'class', [['class1', '\xc2\xae'], ['']])

    def test_invalid_bytes(self):
        # \x80 is invalid utf-8
        s = """<div CLASS='\x80'>text in div</div><p>invalid bytes \x80</p>"""
        blocks = Blockifier.blockify(s, encoding='utf-8')
        self.block_output_tokens(blocks, [['text', 'in', 'div']])
        self.css_output_tokens(blocks, 'class', [['\xc2\x80']])

    def test_big_html(self):
        s = big_html_doc
        blocks = Blockifier.blockify(s)
        self.block_output_tokens(
            blocks,
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
             ['Footer', 'text']]
            )
        self.link_output_tokens(
            blocks,
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
             []]
            )
        self.css_output_tokens(
            blocks, 'class',
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
             ['footer']]
            )
        self.css_output_tokens(
            blocks, 'id',
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
             ['']]
            )


class TestKohlschuetter(KohlschuetterUnitBase):

    def test_small_doc(self):
        kf = KohlschuetterFeatures()

        s = '<html></html>'
        with self.assertRaises(ValueError):
            kf.transform(Blockifier.blockify(s))

        s = '<html> <p>a</p> <div>b</div> </html>'
        with self.assertRaises(ValueError):
            kf.transform(Blockifier.blockify(s))

    def test_transform(self):
        kf = KohlschuetterFeatures()
        s = '<html> <p>first </p> <div> <p>second block with <a href=''>anchor</a> </p> <p>the third block</p> </div> </html>'
        blocks = Blockifier.blockify(s)
        features = kf.transform(blocks)
        self.block_output_tokens(blocks, [['first'], ['second', 'block', 'with', 'anchor'], ['the', 'third', 'block']])
        self.link_output_tokens(blocks, [[], ['anchor'], []])

        text_density = [1.0, 4.0, 3.0]
        link_density = [1.0, 0.25, 1.0 / 3.0]

        self.assertTrue(np.allclose(features[0, :], [0.0, 0.0, link_density[0], text_density[0], link_density[1], text_density[1]]))
        self.assertTrue(np.allclose(features[1, :], [link_density[0], text_density[0], link_density[1], text_density[1], link_density[2], text_density[2]]))
        self.assertTrue(np.allclose(features[2, :], [link_density[1], text_density[1], link_density[2], text_density[2], 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()

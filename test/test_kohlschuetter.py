
import unittest
from dragnet import KohlschuetterBase, PartialBlock, BlockifyError, Kohlschuetter
from lxml import etree
import re
import numpy as np

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

class TestKohlschuetterBase(KohlschuetterUnitBase):
    def test_lxml_error(self):
        """tests the case where lxml raises an error during parsing

        also handles case where lxml returns None for the tree"""
        # this raises an error in parsing
        self.assertRaises(etree.XMLSyntaxError, etree.fromstring, '', etree.HTMLParser(recover=True))
        self.assertRaises(BlockifyError, KohlschuetterBase.blockify, '')

        # this returns None in lxml
        self.assertTrue(None == etree.fromstring('<!--', etree.HTMLParser(recover=True)) )
        self.assertRaises(BlockifyError, KohlschuetterBase.blockify, '<!--')


    def test_very_simple(self):
        """test_very_simple"""
        s = """<div>some text
                    <script> skip this </script>
                    more text here
               </div>"""
        blocks = KohlschuetterBase.blockify(s)
        self.block_output_tokens(blocks, [['some', 'text', 'more', 'text', 'here']])

    def test_very_simple2(self):
        s = """<div>some text <i>in italic</i> and something else
                    <pre> <div>skip this</div> </pre>
                    <b>bold stuff</b> after the script
               </div>"""
        blocks = KohlschuetterBase.blockify(s)
        self.block_output_tokens(blocks, [['some', 'text', 'in', 'italic', 'and', 'something', 'else', 'bold', 'stuff', 'after', 'the', 'script']])

    def test_simple_two_blocks(self):
        s = """<h1>A title <i>with italics</i> and other words</h1>
               some text outside the h1
               <div>a div <span class="test"> with a span </span> more </div>"""
        blocks = KohlschuetterBase.blockify(s)
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
        blocks = KohlschuetterBase.blockify(s)
        self.block_output_tokens(blocks,
                [['h1', 'tag', 'word', 'orphaned', 'text'],
                 ['table', 'data', 'second', 'row', 'final']])

    def test_empty_blocks(self):
        s = """<div> .! </div>
                some text
               <h1> in an h1 </h1>
               <p> ! _ </p>
            """
        blocks = KohlschuetterBase.blockify(s)
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
        blocks = KohlschuetterBase.blockify(s)
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
        blocks = KohlschuetterBase.blockify(s)

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
        blocks = KohlschuetterBase.blockify(s)
        self.block_output_tokens(blocks,
            [['the', 'registered', 'trademark', u'\xae']])
        self.link_output_tokens(blocks,
            [['the', 'registered', 'trademark', u'\xae']])


    def test_all_non_english(self):
        s = u"""<div> <div> \u03b4\u03bf\u03b3 </div> <div> <a href="summer">\xe9t\xe9</a> </div>
         <div> \u62a5\u9053\u4e00\u51fa </div> </div>"""
        blocks = KohlschuetterBase.blockify(s)
        self.block_output_tokens(blocks,
            [[u'\u03b4\u03bf\u03b3'],
            [u'\xe9t\xe9'],
            [u'\u62a5\u9053\u4e00\u51fa']])
        self.link_output_tokens(blocks,
            [[],
             [u'\xe9t\xe9'],
             []])


    def test_text_from_subtree(self):
        s = """<a href=".">WILL <img src="."> THIS PASS <b>THE TEST</b> ??</a>"""
        tree = etree.fromstring(s, etree.HTMLParser(recover=True))
        text_list = PartialBlock._text_from_subtree(tree, tags_exclude=KohlschuetterBase.blacklist)
        text_str = ' '.join([ele.strip() for ele in text_list if ele.strip() != ''])
        self.assertEqual(text_str,
            'WILL THIS PASS THE TEST ??')

    def test_big_html(self):
        s = """
<html>

<body>
<h1>Inside the h1 tag </h1>
<div id="content">
    <b>First line of the content in bold</b>
    <p>A paragraph with <a href="link_target.html">a link</a> and some 

    additional words.

    <p>Second paragraph

    <blockquote>Insert a block quote here</blockquote>

    <div class="image_css" id="image1"><img src="img.jpg"></div>
    
    <p>Some more text after the image
    <h2>An h2 tag just for kicks</h2>
    <p>Finally more text at the end of the content
</div>


<div class="begin_comments">
    <div id="comment1">
        <p>This is a comment</p>
        <p>with two paragraphs <a href="spam_link.html">and some comment spam</a>
    </div>
    <div id="comment2">
        <p>Second comment</p>
    </div>
</div>

<div class="footer"><a href="footer_link.html"><img src="footer_image.jpg" alt="image as anchor text"></a>Footer text
</div>

</html>
"""
        blocks = KohlschuetterBase.blockify(s)

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


class TestKohlschuetter(KohlschuetterUnitBase):
    def test_small_doc(self):
        # need an instance to call analyze
        koh = Kohlschuetter()

        self.assertEqual((None, []), Kohlschuetter.make_features('<html></html>'))
        self.assertEqual('', koh.analyze('<html></html>'))

        s = '<html> <p>a</p> <div>b</div> </html>'
        features, blocks = Kohlschuetter.make_features(s)
        self.assertTrue(features is None)
        self.block_output_tokens(blocks, [['a'], ['b']])
        self.assertEqual('a b', koh.analyze(s))


    def test_make_features(self):
        s = '<html> <p>first </p> <div> <p>second block with <a href=''>anchor</a> </p> <p>the third block</p> </div> </html>'
        features, blocks = Kohlschuetter.make_features(s)
        self.block_output_tokens(blocks, [['first'], ['second', 'block', 'with', 'anchor'], ['the', 'third', 'block']])
        self.link_output_tokens(blocks, [[], ['anchor'], []])

        text_density = [1.0, 4.0, 3.0]
        link_density = [1.0, 0.25, 1.0 / 3.0]

        self.assertTrue(np.allclose(features[0, :], [0.0, 0.0, link_density[0], text_density[0], link_density[1], text_density[1]]))
        self.assertTrue(np.allclose(features[1, :], [link_density[0], text_density[0], link_density[1], text_density[1], link_density[2], text_density[2]]))
        self.assertTrue(np.allclose(features[2, :], [link_density[1], text_density[1], link_density[2], text_density[2], 0.0, 0.0]))





if __name__ == "__main__":
    unittest.main()




import unittest
from dragnet import Blockifier, PartialBlock, BlockifyError, kohlschuetter, ContentExtractionModel, NormalizedFeature, kohlschuetter_features
from lxml import etree
import re
import numpy as np
from mozsci.models import LogisticRegression


# document for testing
big_html_doc = """
<html>

<body>
<h1>Inside the h1 tag </h1>
<div id="content">
    <b class="title">First line of the content in bold</b>
    <p id="para">A paragraph with <a class="link" href="link_target.html">a link</a> and some 

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

    @staticmethod
    def count_divs(tree):
        div_xpath = etree.XPath("//div")
        TestKohlschuetterBase.div_count = len(div_xpath(tree))

    def test_callback(self):
        s = """<div>some text <i>in italic</i> and something else
                    <pre> <div>skip this</div> </pre>
                    <b>bold stuff</b> after the script
               </div>"""
        blocks = KohlschuetterBase.blockify(s,
                                            TestKohlschuetterBase.count_divs)
        self.assertEqual(TestKohlschuetterBase.div_count, 2)

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




    def test_text_from_subtree(self):
        s = """<a href=".">WILL <img src="."> THIS PASS <b>THE TEST</b> ??</a>"""
        tree = etree.fromstring(s, etree.HTMLParser(recover=True))
        text_list = PartialBlock._text_from_subtree(tree, tags_exclude=Blockifier.blacklist)
        text_str = ' '.join([ele.strip() for ele in text_list if ele.strip() != ''])
        self.assertEqual(text_str,
            'WILL THIS PASS THE TEST ??')



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



class TestDragnetModelKohlschuetterFeatures(unittest.TestCase):
    def test_dragnet_model(self):
        params = {'b':0.2, 'w':[0.4, -0.2, 0.9, 0.8, -0.3, -0.5]}
        block_model = LogisticRegression.load_model(params)
        mean_std = {'mean':[0.0, 0.1, 0.2, 0.5, 0.0, 0.3], 'std':[1.0, 2.0, 0.5, 1.2, 0.75, 1.3]}
        koh_features = NormalizedFeature(kohlschuetter_features, mean_std)
    
        dm = ContentExtractionModel(Blockifier, [koh_features], block_model, threshold=0.5)
        content = dm.analyze(big_html_doc)
    
        # make prediction from individual components
        # this assumes:  kohlschuetter.make_features and uses LogisticRegression
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



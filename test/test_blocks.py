import io
import os

import pytest
from lxml import etree

from dragnet import blocks


@pytest.fixture(scope="module")
def html1():
    fname = os.path.join("test", "datafiles", "HTML", "page_for_testing.html")
    with io.open(fname, mode="rt") as f:
        html_ = f.read()
    return html_


@pytest.fixture(scope="module")
def html2():
    return """
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
        """


def test_guess_encoding():
    s = b"""<?xml version="1.0" encoding="ISO-8859-1"?>
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN"
      "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">

      <html xmlns="http://www.w3.org/1999/xhtml" xml:lang="fr" lang="fr">
    """
    assert blocks.guess_encoding(s) == b'ISO-8859-1'

    s = b"""<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN"
      "http://www.w3.org/TR/html4/strict.dtd">

      <head>
      <meta http-equiv="content-type" content="text/html; charset=GB2312">
      </head>
    """
    assert blocks.guess_encoding(s) == b'GB2312'

    s = b"<html>sadfsa</html>"
    assert blocks.guess_encoding(s, b'asciI') == b'asciI'


class TestTextSubtree(object):

    def test_text_from_subtree(self):
        s = '<a href=".">WILL <img src="."> THIS PASS <b>THE TEST</b> ??</a>'
        tree = etree.fromstring(s, etree.HTMLParser(recover=True))
        text_list = blocks.text_from_subtree(tree)
        text_str = " ".join([ele.strip() for ele in text_list if ele.strip() != ""])
        assert text_str == "WILL THIS PASS THE TEST ??"

    def test_text_from_subtree_decode_error(self):
        # this is an invalid utf-8 character
        s = b"<div>\x92</div>"
        tree = etree.fromstring(s, etree.HTMLParser(recover=True, encoding="utf-8"))
        text_list = blocks.text_from_subtree(tree)
        text_str = " ".join([ele.strip() for ele in text_list if ele.strip() != ""])
        assert text_str == ""


def check_tagcount(expected, predicted):
    assert predicted['tagcount'] == expected[0]
    assert predicted['tagcount_since_last_block'] == expected[1]
    assert predicted['anchor_count'] == expected[2]
    assert predicted['min_depth_since_last_block'] == expected[3]


class TestTagCountPB(object):

    def test_simple(self):
        s = """<html><body><div>some text <i>in italic</i> and something else
                    <script> <div>skip this</div> </script>
                    <b>bold stuff</b> after the script
               </div></body></html>"""
        blks = blocks.TagCountBlockifier.blockify(s)
        check_tagcount((3, 2, 0, 0), blks[0].features)
        assert len(blks) == 1

    def test_big_html(self, html1):
        blks = blocks.TagCountBlockifier.blockify(html1)
        actual_features = [
            (1, 2, 0, 0),
            (2, 0, 0, 2),
            (2, 0, 1, 3),
            (2, 0, 0, 3),  # blockquote
            (1, 2, 0, 3),
            (1, 0, 0, 3),
            (1, 0, 0, 3),
            (1, 2, 0, 2),  # first comment
            (2, 0, 1, 4),
            (1, 1, 0, 3),
    #            (3, 0, 1, 0)  # NOTE: this is a bug here.  It's due
                    # to the _tc-1 assumption in the feature extractor
                    # that fails for the last block. (we don't call
                    # tag_tagcount again before appending the block)
        ]
        for a, b in zip(actual_features, blks):
            check_tagcount(a, b.features)


class TestReadabilityBlocks(object):

    def test_ancestors(self, html2):
        blks = blocks.TagCountReadabilityBlockifier.blockify(html2)
        # get the text and ancestors from the blocks
        actual = [(blk.text, blk.features['ancestors']) for blk in blks]
        expected = [
            ('1 i', [0, 2]),
            ('2', [0, 2, 4]), ('3', [0, 2, 4]), ('4', [0, 2, 4]),
            ('5', [0, 2, 4, 9]), ('6', [0, 2, 4, 9]),
            ('7', [0, 2])]
        assert actual == expected

    def test_class_weights(self, html2):
        blks = blocks.TagCountReadabilityBlockifier.blockify(html2)
        actual = [blk.features['readability_class_weights'] for blk in blks]
        expected = [
            [(0, 0), (2, 0), (4, 30), (6, 0)], [(7, -25)], [(8, 0)],
            [(9, -20)], [(11, 0)], [(12, 0)], [(13, 5), (14, -5)]
        ]
        assert actual == expected

    def test_block_start_tag(self, html2):
        blks = blocks.TagCountReadabilityBlockifier.blockify(html2)
        actual = [blk.features['block_start_tag'] for blk in blks]
        expected = ['div', 'p', 'p', 'div', 'p', 'p', 'h1']
        assert actual == expected


def _run_blockifier_whitespace_test(html, expected_text):
    parsed_blocks = blocks.Blockifier.blockify(html)
    parsed_text = "\n".join([b.text for b in parsed_blocks])
    assert parsed_text.replace("\u2028", "\n") == expected_text


class TestBlockifierWhitespace:
    def test_collapse_whitespace_simple(self):
        html = """<html><body><p>paragraph</p><div>   A simple    test   <a href="url">with a    link</a>.  \n
                </div></body></html>"""
        expected_text = "paragraph\nA simple test with a link."
        _run_blockifier_whitespace_test(html, expected_text)

    def test_collapse_whitespace_2(self):
        html = """<html><body><div>A<a href="url">anchor <i>text</i>1</a> .  </div></body></html>"""
        expected_text = "Aanchor text1 ."
        _run_blockifier_whitespace_test(html, expected_text)

    def test_collapse_whitespace_3(self):
        html = """<html><body><div>A<a href="url">anchor <i>text</i> 1</a>.  </div></body></html>"""
        expected_text = "Aanchor text 1."
        _run_blockifier_whitespace_test(html, expected_text)

    def test_collapse_whitespace_nbsp(self):
        html = """<html><body><div>  Some     text&nbsp;as a    test  &nbsp;case&nbsp;&nbsp;?
            </div>      <div>more&ensp;and more&emsp;spaces</div></body></html>"""
        expected_text = 'Some text\xa0as a test \xa0case\xa0\xa0?\nmore\u2002and more\u2003spaces'
        _run_blockifier_whitespace_test(html, expected_text)

    def test_br_whitespace(self):
        html = """
        <html><body>
        <div>Text first div</div>
        <div>Second <div>nested div</div>  with br  <br>New line<br> another </div>
        </body></html>
        """
        expected_text = "Text first div\nSecond\nnested div\nwith br\nNew line\nanother"
        _run_blockifier_whitespace_test(html, expected_text)

    def test_p_whitespace(self):
        html = """
        <html><body>
        <div>Text first div</div>

        <p></p>

        Something <br> here
        </body></html>
        """
        expected_text = 'Text first div\n\nSomething\nhere'
        _run_blockifier_whitespace_test(html, expected_text)

    def test_p_whitespace_2(self):
        html = """
        <html><body>
        <div>Text first div</div>

        <p></p>

        <div> Something <div> <br> here <br><br> <br>&nbsp;<br>end</div>
        </body></html>
        """
        expected_text = "Text first div\nSomething\n\nhere\n\n\n\xa0\nend"
        _run_blockifier_whitespace_test(html, expected_text)

    def test_whitespace_pre(self):
        html = '<div>First text <pre>  Pre <div>    formatted\n        <a href="url">link   </a>.  </div> after  .  the</pre>   between      </div> last    <pre>   last pre </pre>'
        expected_text = 'First text\n\n  Pre \n    formatted\n        link   .  \n after  .  the\n\n\nbetween\nlast\n\n   last pre \n'
        _run_blockifier_whitespace_test(html, expected_text)

#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implementation of the blockifier interface and some classes
to manipulate blocks

blockifier interface is any object that implements blockify(html) and
returns a list of Block instances
"""

# cython imports
from libcpp.set cimport set as cpp_set
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map as cpp_map
from libcpp cimport bool
from cython.operator cimport preincrement as inc
from cython.operator cimport dereference as deref

# boilerplate from http://lxml.de/capi.html
cimport etreepublic as cetree
cdef object etree
from lxml import etree
cetree.import_lxml__etree()

# python imports
import re
import numpy as np
import math

from itertools import chain


re_tokenizer = re.compile('[\W_]+', re.UNICODE)
simple_tokenizer = lambda x: [ele for ele in re_tokenizer.split(x) if len(ele) > 0]

# need a typedef for the callback function in text_from_subtree
# and a default function that does nothing
# http://stackoverflow.com/questions/14124049/is-there-any-type-for-function-in-cython
ctypedef void (*callback_t)(PartialBlock, cetree.tree.xmlNode*)
cdef void empty_callback(PartialBlock pb, cetree.tree.xmlNode* x):
    return

# typedefs for the functions that subclasses of PartialBlock implement
ctypedef void (*reinit_t)()
ctypedef cpp_map[string, int] (*name_t)(bool)
ctypedef void (*subtree_t)(int)


# tags we'll ignore completely
cdef cpp_set[string] BLACKLIST
BLACKLIST = set([
    'applet', 'area', 'base', 'basefont', 'bdo', 'button', 
    'caption', 'fieldset', 'fram', 'frameset', 
    'iframe', 'img', 'input', 'legend', 'link', 'menu', 'meta', 
    'noframes', 'noscript', 'object', 'optgroup', 'option', 'param', 
    'script', 'select', 'style', 'textarea', 'var', 'xmp',
    'like', 'like-box', 'plusone'
])


# tags defining the blocks we'll extract
cdef cpp_set[string] BLOCKS
BLOCKS = set([
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'table', 'map',
])


class Block(object):
    def __init__(self, text, link_density, text_density,
            anchors, link_tokens, css, **kwargs):
        self.text = text
        self.link_density = link_density
        self.text_density = text_density
        self.anchors = anchors
        self.link_tokens = link_tokens  # a hook for testing
        self.css = css
        self.features = kwargs


class BlockifyError(Exception):
    """Raised when there is a fatal problem in blockify
    (if lxml fails to parse the document)
    """
    pass


cdef vector[string] _text_from_subtree(cetree.tree.xmlNode *tree,
    bool tail, callback_t callback, PartialBlock klass):
    '''
    A faster, Cython version of text_from_subtree
    '''

    cdef vector[string] text
    text.clear()
    cdef object t

    try:
        t = cetree.textOf(tree)
        if t is not None:
            text.push_back(t.encode('utf-8'))
    except UnicodeDecodeError:
        pass


    cdef cetree.tree.xmlNode *node, *next_node
    cdef string tag
    cdef vector[string] to_add
    cdef size_t k

    if cetree.hasChild(tree):
        node = cetree.findChild(tree, 0)
    else:
        node = NULL

    while node != NULL:

        # call the feature extractor child hooks
        callback(klass, node)

        # get tag, check whether in black list
        tag = cetree.namespacedName(node)
        if BLACKLIST.find(tag) == BLACKLIST.end():
            to_add = _text_from_subtree(node, True, callback, klass)
            for k in range(to_add.size()):
                text.push_back(to_add[k])
        else:
            # get the tail
            try:
                t = cetree.tailOf(node)
                if t is not None:
                    text.push_back(t.encode('utf-8'))
            except UnicodeDecodeError:
                pass

        next_node = cetree.nextElement(node)
        node = next_node

    if tail:
        try:
            t = cetree.tailOf(tree)
            if t is not None:
                text.push_back(t.encode('utf-8'))
        except UnicodeDecodeError:
            pass

    return text


#def run_me():
#    # TODO - delete this!
#    cdef cetree._Element tt
#
#    s = """<a href=".">WILL <img src="."> THIS PASS <b>THE TEST</b> ??</a>"""
#    tree = etree.fromstring(s, etree.HTMLParser(recover=True))
#    tt = tree
#    print _text_from_subtree(tt._c_node, True, empty_callback)
#
#    s = '<div>\x92</div>'
#    tree = etree.fromstring(s, etree.HTMLParser(recover=True, encoding='utf-8'))
#    tt = tree
#    print _text_from_subtree(tt._c_node, True, empty_callback)
#
#    s = '''<div>some words <-- with a comment --></div> <!-- another comment! -->
#    <script>SKIP THIS!</script>'''
#    tree = etree.fromstring(s, etree.HTMLParser(recover=True, encoding='utf-8'))
#    tt = tree
#    print _text_from_subtree(tt._c_node, True, empty_callback)
#


def text_from_subtree(tree, tags_exclude=set(), tail=True, callback=None):
    """Get all the text
    from the subtree, excluding tags_exclude
    If tail=False, then don't append the tail for this top level element
    callbacks = called with callback(child) when iterating through the tree"""
    try:
        text = [tree.text or '']
    except UnicodeDecodeError:
        text = []
    for child in tree.iterchildren():

        # call the feature extractor child hooks
        if callback:
            callback(child)

        if child.tag not in tags_exclude:
            text.extend(text_from_subtree(child, tags_exclude=tags_exclude, callback=callback))
        else:
            # get the tail
            try:
                text.append(child.tail or '')
            except UnicodeDecodeError:
                pass
    if tail:
        try:
            text.append(tree.tail or '')
        except UnicodeDecodeError:
            pass
    return text


cdef class PartialBlock:
    """As we create blocks by recursing through subtrees
    in Blockifier, we need to maintain some state
    of the incomplete blocks.

    This class maintains that state, as well as provides methods
    to modify it.

    To generalize, subclasses do the following:
        define a set of "feature extractors".  These are given
        a string name and specified by the following methods:

        reinit_name() = r
        name(self) = compute the features as a dict
                    that is passed into the Block constructor, called
                    just before the block is added to the results
        tag_name(self, child) = called with each tag as we iterate through
            the tree
        subtree_name(1) called once before iterating over each subtree
        subtree_name(-1) called once after iterating over the subtree

    Can specify a set of call backs that are called with
    the partial block instance just before the block is added
    to the results.  These compute the features as a dict
    that is passed into the Block constructor.  These are implemented
    by subclasses.
    """
    cdef vector[string] css_attrib
    cdef vector[string] text
    cdef vector[string] link_tokens
    cdef list anchors
    cdef cpp_map[string, vector[string]] css_tree
    cdef cpp_map[string, vector[string]] css

    # subclass callbacks
    cdef vector[callback_t] _tag_func
    cdef vector[reinit_t] _reinit_func
    cdef vector[name_t] _name_func
    cdef vector[subtree_t] _subtree_func

    def __cinit__(self):
        self.css_attrib.clear()
        self.css_attrib.push_back('id')
        self.css_attrib.push_back('class')

    def __init__(self):
        self._tag_func.clear()
        self._reinit_func.clear()
        self._name_func.clear()
        self._subtree_func.clear()
        self.reinit()
        self.reinit_css(True)

    cdef _fe_reinit(self):
        # each subclass implements reinit_fename()
        # call self.reinit_name() for each name
        cdef size_t k
        for k in range(self._reinit_func.size()):
            self._reinit_func[k]()

    cdef reinit(self):
        self.text.clear()
        self.link_tokens.clear()
        self.anchors = []
        self._fe_reinit()

    cdef reinit_css(self, bool init_tree):
        # we want to keep track of the id and class CSS attributes.
        # we will accumulate a few sources of them
        # (1) the CSS attributes for the current tag and all tags in trees containing
        #     the current one
        # (2) the CSS attributes for all tags inside this block

        # css_tree will hold the attributes for all trees containing this tag
        # css will hold the accumulated attributes for this block,
        #   and will be initialized with the tree

        cdef size_t k
        if init_tree:
            # the initial init
            self.css_tree.clear()
            self.css.clear()
            for k in range(self.css_attrib.size()):
                self.css_tree[self.css_attrib[k]].clear()
                self.css[self.css_attrib[k]].clear()
        else:
            # we are re-initializing after creating a block
            # css_tree is unchanged and css is set to css_tree
            self.css.clear()
            for k in range(self.css_attrib.size()):
                self.css[self.css_attrib[k]].clear()

    cdef cpp_map[string, int] _extract_features(self, bool append):
        # call self.fe_name(append=True/False) where
        # append is True if this PartialBlock is appended
        # or False if it is not.
        cdef cpp_map[string, int] ret
        cdef cpp_map[string, int] to_add
        cdef cpp_map[string, int].iterator it
        cdef size_t k
        ret.clear()
        for k in range(self._name_func.size()):
            to_add = self._name_func[k](append)
            it = to_add.begin()
            while it != to_add.end():
                ret[deref(it).first] = deref(it).second
                inc(it)
        return ret


    cdef add_block_to_results(self, list results):
        """Create a block from the current partial block
        and append it to results.  Reset the partial block"""

        # compute block and link tokens!
        block_tokens = self.tokens_from_text(self.text)
        cdef size_t k
        cdef string cssa
        if len(block_tokens) > 0:
            # only process blocks with something other then white space
            block_text = ' '.join(block_tokens)
            link_text = ' '.join(self.tokens_from_text(self.link_tokens))

            # compute link/text density
            link_d = self.link_density(block_text, link_text)
            text_d = self.text_density(block_text)

            # get the id, class attributes
            css = {}
            for k in range(self.css_attrib.size()):
                cssa = self.css_attrib[k]
                css[cssa] = ' '.join(
                    self.tokens_from_text(self.css[cssa])).lower()

            results.append(Block(block_text, link_d,
                        text_d, self.anchors, self.link_tokens, css,
                        **self._extract_features(True)))
        else:
            self._extract_features(False)

        self.reinit()
        self.reinit_css(False)

    cdef add_text(self, cetree.tree.xmlNode *ele, string text_or_tail):
        """Add the text/tail from the element
        text_or_tail is 'text' or 'tail'"""
        cdef object t
        try:
            if text_or_tail == <string>'text':
                t = cetree.textOf(ele)
            else:
                t = cetree.tailOf(ele)
            if t is not None:
                self.text.push_back(t.encode('utf-8'))
        except UnicodeDecodeError:
            pass


    cdef add_anchor(self, cetree.tree.xmlNode* ele, cetree._Document doc):
        """Add the anchor tag to the block"""
        self.anchors.append(cetree.elementFactory(doc, ele))
        # need all the text from the subtree

        # NOTE: here we don't worry about calling _subtree_fe inside
        # text_from_subtree, even though it is recursive.
        # this is because we will never output a block in the middle of
        # these subtrees so we don't need to keep track of going in and
        # then out

        cdef vector[string] anchor_text_list

        anchor_text_list = _text_from_subtree(ele, False, self._tag_fe, self)

        cdef size_t k
        for k in range(anchor_text_list.size()):
            self.text.push_back(anchor_text_list[k])

        cdef object t
        try:
            t = cetree.tailOf(ele)
            if t is not None:
                self.text.push_back(t.encode('utf-8'))
        except UnicodeDecodeError:
            pass

        cdef vector[string] anchor_tokens
        anchor_tokens = self.tokens_from_text(anchor_text_list)
        for k in range(anchor_tokens.size()):
            self.link_tokens.push_back(anchor_tokens[k])


    cdef void _tag_fe(self, cetree.tree.xmlNode* child):
        # call the tag_featurename functions
        cdef size_t k
        for k in range(self._tag_func.size()):
            self._tag_func[k](self, child)

    cdef void _subtree_fe(self, int start_or_end):
        # call the subtree_featurename functions
        cdef size_t k
        for k in range(self._subtree_func.size()):
            self._subtree_func[k](start_or_end)
            
    cdef tokens_from_text(self, text):
        """given a list of text (as built in self.text for example)
        return a list of tokens.

        according to 
        http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
        chain is the fastest way to combine the list of lists"""
        return list(chain.from_iterable([re.split('\s+', ele.strip()) for ele in text if ele.strip() != '']))


    cdef link_density(self, block_text, link_text):
        '''
        Assuming both input texts are stripped of excess whitespace, return the 
        link density of this block
        '''
        # NOTE: in the case where link_text == '', this re.split
        # returns [''], which incorrectly
        # has 1 token by it's length instead of 0
        # however, fixing this bug decreases model performance by about 1%,
        # so we keep it
        anchor_tokens = re_tokenizer.split(link_text)
        block_tokens = re_tokenizer.split(block_text)
        return float(len(anchor_tokens)) / len(block_tokens)


    cdef text_density(self, block_text):
        '''
        Assuming the text has been stripped of excess whitespace, return text
        density of this block
        '''
        lines = math.ceil(len(block_text) / 80.0)

        if int(lines) == 1:
            tokens = re_tokenizer.split(block_text)
            return float(len(tokens))
        else:
            # need the number of tokens excluding the last partial line
            tokens = re_tokenizer.split(block_text[:(int(lines - 1) * 80)])
            return len(tokens) / (lines - 1.0)


    cdef recurse(self, cetree.tree.xmlNode* subtree, list results,
        cetree._Document doc):

        cdef cetree.tree.xmlNode *node, *next_node
        cdef string tag

        # for CSS, we want to output all CSS tags for all levels in subtree
        # we will add them on entry, and pop them on exit
        self.update_css(subtree, True)

        self._subtree_fe(1)

        # iterate through children
        if cetree.hasChild(subtree):
            node = cetree.findChild(subtree, 0)
        else:
            node = NULL

        while node != NULL:

            if self._tag_func.size() > 0:
                self._tag_fe(node)

            # get tag
            tag = cetree.namespacedName(node)

            if BLACKLIST.find(tag) != BLACKLIST.end():
                # in the blacklist
                # in this case, skip the entire tag,
                # but it might have some tail text we need
                self.add_text(node, <string>'tail')

            elif BLOCKS.find(tag) != BLOCKS.end():
                # this is the start of a new block
                # add the existing block to the list,
                # start the new block and recurse
                self.add_block_to_results(results)
                self.add_text(node, <string>'text')
                self.update_css(node, False)
                self.recurse(node, results, doc)
                self.add_text(node, <string>'tail')

            elif tag == <string>'a':
                # an anchor tag
                self.add_anchor(node, doc)
                self.update_css(node, False)

            else:
                # a standard tag.
                # we need to get it's text and then recurse over the subtree
                self.add_text(node, <string>'text')
                self.update_css(node, False)
                self.recurse(node, results, doc)
                self.add_text(node, <string>'tail')

            # reset for next iteration
            next_node = cetree.nextElement(node)
            node = next_node

        self.pop_css_tree()
        self._subtree_fe(-1)

    cdef update_css(self, cetree.tree.xmlNode *child, bool tree):
        """Add the child's tag to the id, class lists"""
        cdef cpp_map[string, vector[string]] *css_to_update
        if tree:
            css_to_update = &(self.css_tree)
        else:
            css_to_update = &(self.css)

        cdef size_t k
        cdef string attrib
        cdef cetree.tree.xmlAttr* attr

        for k in range(self.css_attrib.size()):
            attrib = self.css_attrib[k]
            attr = cetree.tree.xmlHasProp(child,
                <cetree.tree.const_xmlChar*> attrib.c_str())
            if attr is NULL:
                deref(css_to_update)[attrib].push_back(<string>'')
            else:
                deref(css_to_update)[attrib].push_back(
                    <string>cetree.attributeValue(child, attr).encode('utf-8'))


    cdef pop_css_tree(self):
        """pop the last entry off the css lists"""
        cdef size_t k
        for k in range(self.css_attrib.size()):
            if self.css_tree[self.css_attrib[k]].size() > 0:
                self.css_tree[self.css_attrib[k]].pop_back()


#class TagCountPB(PartialBlock):
#    """Counts tags to compute content-tag ratios"""
#
#    # Associate with each block a tag count = the count of tags
#    #   in the block.
#    # Since we don't output empty blocks, we also keep track of the
#    # tag count since the last block we output as an additional feature
#    #
#    
#    # _tc = tag count in the current block, since the last <div>, <p>, etc.
#    # _tc_lb = tag count since last block.  This is the tag count in prior
#    # empty blocks, accumulated since the last block was output, excluding
#    # the current block
#    
#    # so tc gets updated with each tag
#    # tc is reset on block formation, even for empty blocks
#    #
#    # tc_lb is reset to 0 on block output
#    # tc_lb accumulates tc on empty block formation
#    #
#
#    def __init__(self, *args, **kwargs):
#        PartialBlock.__init__(self, *args, **kwargs)
#
#        self._fe.append('tagcount')
#
#        # will keep track of tag count and tag count since last block
#        self._tc = 1  # for the top level HTML tag
#        self._ac = 0  # anchor count
#        self._tc_lb = 0
#        self._current_depth = 0
#        self._min_depth_last_block = 0
#        self._min_depth_last_block_pending = 0
#
#    def reinit_tagcount(self):
#        pass
#
#    def subtree_tagcount(self, start_or_end):
#        self._current_depth += start_or_end
#        self._min_depth_last_block_pending = min(self._min_depth_last_block_pending, self._current_depth)
#
#    def tagcount(self, append):
#        # here we assume that tc is updated
#        # before features are extracted
#        # that is, _tc includes the current tag
#        # so we adjust by -1 on output
#        #
#        # since tc has already been updated
#        if append:
#            ret = {'tagcount_since_last_block':self._tc_lb,
#                   'tagcount':self._tc - 1,
#                   'anchor_count':self._ac,
#                   'min_depth_since_last_block':self._min_depth_last_block}
#            self._tc_lb = 0
#            self._tc = 1
#            self._ac = 0
#            self._min_depth_last_block_pending = self._current_depth
#            self._min_depth_last_block = self._current_depth
#        else:
#            ret = {}
#            self._tc_lb += (self._tc - 1)
#            self._tc = 1
#            self._ac = 0
#        return ret
#
#    def tag_tagcount(self, tag):
#        self._tc += 1
#        if tag.tag == 'a':
#            self._ac += 1
#
#        if tag.tag not in Blockifier.blocks:
#            self._min_depth_last_block = self._min_depth_last_block_pending
#            
##        print tag.tag, self._tc, self._tc_lb



html_re = re.compile('meta\s[^>]*charset\s*=\s*"{0,1}\s*([a-zA-Z0-9-]+)', re.I)
xml_re = re.compile('<\?\s*xml[^>]*encoding\s*=\s*"{0,1}\s*([a-zA-Z0-9-]+)', re.I)
def guess_encoding(s, default='utf-8'):
    """Try to guess the encoding of s -- check the XML declaration
    and the HTML meta tag
    
    if default=CHARDET then use chardet to guess the default"""
    mo = xml_re.search(s)
    if mo:
        encoding = mo.group(1)
    else:
        moh = html_re.search(s)
        if moh:
            encoding = moh.group(1)
        else:
            if default == 'CHARDET':
                from chardet.universaldetector import UniversalDetector
                u = UniversalDetector()
                u.feed(s)
                u.close()
                encoding = u.result['encoding']
                print "Guessing encoding with chardet of %s" % encoding
            else:
                encoding = default
    return encoding


class TagCountBlockifier:
    pass

class Blockifier(object):
    """A blockifier for web-page de-chroming that loosely follows the approach in
        Kohlschütter et al.:
        http://www.l3s.de/~kohlschuetter/publications/wsdm187-kohlschuetter.pdf

      Implements the blockify interface
    """

    @staticmethod
    def blocks_from_tree(tree, pb=PartialBlock):
        cdef list results = []
        cdef cetree._Element ctree

        cdef PartialBlock partial_block = pb()
        ctree = tree
        partial_block.recurse(ctree._c_node, results, ctree._doc)

        # make the final block
        partial_block.add_block_to_results(results)

        return results
    

    @staticmethod
    def blockify(s, encoding=None, pb=PartialBlock):
        '''
        Take a string of HTML and return a series of blocks

        if encoding is None, then try to extract it from the HTML
        '''
        # First, we need to parse the thing
        encoding = encoding or guess_encoding(s, default='utf-8')
        try:
            html = etree.fromstring(s,
                etree.HTMLParser(recover=True, encoding=encoding,
                remove_comments=True, remove_pis=True))
        except:
            raise BlockifyError
        if html is None:
            # lxml sometimes doesn't raise an error but returns None
            raise BlockifyError

        blocks = Blockifier.blocks_from_tree(html, pb)
        # only return blocks with some text content
        return [ele for ele in blocks if re_tokenizer.sub('', ele.text) != '']


#class TagCountBlockifier(Blockifier):
#    @staticmethod
#    def blockify(s, encoding=None):
#        return Blockifier.blockify(s, encoding, pb=TagCountPB)


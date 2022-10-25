"""
Implementation of the blockifier interface and some classes
to manipulate blocks

blockifier interface is any object that implements blockify(html) and
returns a list of Block instances
"""

# cython imports
cimport cython
from libcpp.set cimport set as cpp_set
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map as cpp_map
from libcpp.pair cimport pair
from libcpp cimport bool
from cython.operator cimport preincrement as inc
from cython.operator cimport dereference as deref
from libc.stdint cimport uint32_t

# boilerplate from http://lxml.de/capi.html
cimport lxml.includes.etreepublic as cetree
cdef object etree
from lxml import etree
cetree.import_lxml__etree()

# python imports
import re
import numpy as np
import math

from compat import str_list_cast, str_dict_cast, str_block_cast, str_block_list_cast, str_cast, bytes_cast

RE_HTML_ENCODING = re.compile(
    b'<\s*meta[^>]+charset\s*?=\s*?[\'"]?([^>]*?)[ /;\'">]',
    flags=re.IGNORECASE)
RE_XML_ENCODING = re.compile(
    b'^<\?.*?encoding\s*?=\s*?[\'"](.*?)[\'"].*?\?>',
    flags=re.IGNORECASE)
RE_TEXT = re.compile(r'[^\W_]+', flags=re.UNICODE)
re_tokenizer = re.compile(r'[\W_]+', re.UNICODE)
re_tokenizer_nounicode = re.compile(b'[\W_]+')


def simple_tokenizer(x):
    return [ele for ele in re_tokenizer.split(x) if len(ele) > 0]

# need a typedef for the callback function in text_from_subtree
# and a default function that does nothing
# http://stackoverflow.com/questions/14124049/is-there-any-type-for-function-in-cython
ctypedef void (*callback_t)(PartialBlock, string)
cdef void empty_callback(PartialBlock pb, string x):
    return

# typedefs for the functions that subclasses of PartialBlock implement
ctypedef void (*reinit_t)(PartialBlock)
ctypedef cpp_map[string, int] (*name_t)(PartialBlock, bool)
ctypedef void (*subtree_t)(PartialBlock, int)

cdef inline int int_min(int a, int b): return a if a <= b else b

# tags we'll ignore completely
cdef cpp_set[string] IGNORELIST
IGNORELIST = {
    b'applet', b'area', b'base', b'basefont', b'bdo', b'button',
    b'caption', b'fieldset', b'fram', b'frameset',
    b'iframe', b'img', b'input', b'legend', b'link', b'menu', b'meta',
    b'noframes', b'noscript', b'object', b'optgroup', b'option', b'param',
    b'script', b'select', b'style', b'textarea', b'var', b'xmp',
    b'like', b'like-box', b'plusone',
    # HTML5 vector image tags and math tags
    b'svg', b'math'
    }

# tags defining the blocks we'll extract
cdef cpp_set[string] BLOCKS
BLOCKS = {b'h1', b'h2', b'h3', b'h4', b'h5', b'h6', b'p', b'div', b'table', b'map'}

# define some commonly used strings here, otherwise Cython will always add
# a little python overhead when using them even though they are constant
cdef string CTEXT = <string>'text'
cdef string CTAIL = <string>'tail'
cdef string A = <string>'a'
cdef string BR = <string>'br'
cdef string NEWLINE = <string>'\n'
cdef string TAGCOUNT_SINCE_LAST_BLOCK = <string>'tagcount_since_last_block'
cdef string TAGCOUNT = <string>'tagcount'
cdef string ANCHOR_COUNT = <string>'anchor_count'
cdef string MIN_DEPTH_SINCE_LAST_BLOCK = <string>'min_depth_since_last_block'


# for the class/id readability score
re_readability_negative = re.compile('combx|comment|com-|contact|foot|footer|footnote|masthead|media|meta|outbrain|promo|related|scroll|shoutbox|sidebar|sponsor|shopping|tags|tool|widget', re.I)
re_readability_positive = re.compile('article|body|content|entry|hentry|main|page|pagination|post|text|blog|story', re.I)

cdef string DIV = <string>'div'

cdef cpp_set[string] READABILITY_PLUS3
READABILITY_PLUS3 = {b'pre', b'td', b'blockquote'}

cdef cpp_set[string] READABILITY_MINUS3
READABILITY_MINUS3 = {b'address', b'ol', b'ul', b'dl', b'dd', b'dt', b'li', b'form'}

cdef cpp_set[string] READABILITY_MINUS5
READABILITY_MINUS5 = {b'h1', b'h2', b'h3', b'h4', b'h5', b'h6', b'th'}


cdef cpp_set[char] WHITESPACE = set([<char>' ', <char>'\t', <char>'\n',
    <char>'\r', <char>'\f', <char>'\v'])

cdef vector[string] _tokens_from_text(vector[string] text, bool remove_whitespace):
    '''
    Given a vector of text, return a vector of individual tokens
    '''
    cdef size_t i, j, start
    cdef bool token
    cdef vector[string] ret

    if remove_whitespace:
        for i in range(text.size()):
            token = False
            for j in range(text[i].length()):
                if WHITESPACE.find(text[i][j]) == WHITESPACE.end():
                    # current char is not whitespace
                    if not token:
                        token = True
                        start = j
                else:
                    # a white space character
                    if token:
                        # write out token
                        ret.push_back(text[i].substr(start, j - start))
                        token = False
            # check last token
            if token:
                ret.push_back(text[i].substr(start, text[i].length() - start))
    else:
        # Just keep everything.
        ret.push_back(b''.join(text))

    return ret


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

        # get the tag
        tag = <string> cetree.namespacedName(node).encode('utf-8')

        # call the feature extractor child hooks
        callback(klass, tag)

        # check whether in black list
        if IGNORELIST.find(tag) == IGNORELIST.end():
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

    The recurse method works as follows:
        given a root node, iterate over the children
        when we encounter <div>, <p>, etc, start a new block.
            when starting a new block do the following:
                call name(self) with the partialblock instance just
                    before creating the block. this returns a key->int
                    map that is added to the block
                after creating the block and adding to output, call reinit()
                    to reset the partialblock

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

    # the tag that starts the block (div, p, h1, etc)
    cdef string block_start_tag
    cdef object block_start_element

    # subclass callbacks
    cdef vector[callback_t] _tag_func
    cdef vector[reinit_t] _reinit_func
    cdef vector[name_t] _name_func
    cdef vector[subtree_t] _subtree_func

    cdef bool do_css
    cdef bool do_readability
    cdef bool do_all_whitespace

    # nodes are identified by a tag_id.  this tracks the id of the current
    # node in the recursion
    cdef uint32_t tag_id
    # the ID for next unseen node
    cdef uint32_t next_tag_id
    # during node recursion, this will maintain a list of all ancestors of
    # current node
    cdef vector[uint32_t] ancestors
    # this stores the ancestor list when a block is created so it can
    # be written when the next block is stored
    cdef vector[uint32_t] ancestors_write
    # for each tag_id, stores whether this weight has been calculated
    # and written out yet
    cdef cpp_set[uint32_t] class_weights_written
    # the class weight is only computed once, when we first see the node
    # we'll keep a list here of all the values ids to write out
    # the first time we see them
    cdef vector[pair[uint32_t, int] ] class_weights

    def __cinit__(self, *args, **kwargs):
        self.css_attrib.clear()
        self.css_attrib.push_back('id')
        self.css_attrib.push_back('class')

    def __init__(self, do_css=True, do_readability=False, do_all_whitespace=False):
        self._tag_func.clear()
        self._reinit_func.clear()
        self._name_func.clear()
        self._subtree_func.clear()
        self.reinit()
        self.reinit_css(True)
        self.do_css = do_css

        self.block_start_tag = b''
        self.block_start_element = None

        self.do_readability = do_readability
        self.ancestors.clear()
        self.ancestors_write.clear()
        self.tag_id = 0
        self.next_tag_id = 1
        self.class_weights_written.clear()
        self.class_weights.clear()
        if do_readability:
            self._subtree_func.push_back(
                <subtree_t>PartialBlock.subtree_readability)
            self._reinit_func.push_back(
                <reinit_t>PartialBlock.reinit_readability)

        self.do_all_whitespace = do_all_whitespace

    cdef void _fe_reinit(self):
        # each subclass implements reinit_fename()
        # call self.reinit_name() for each name
        cdef size_t k
        for k in range(self._reinit_func.size()):
            self._reinit_func[k](self)

    cdef void reinit(self):
        self.text.clear()
        self.link_tokens.clear()
        self.anchors = []
        self._fe_reinit()

    cdef void reinit_css(self, bool init_tree):
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

    cdef object _extract_features(self, bool append):
        # call self.fe_name(append=True/False) where
        # append is True if this PartialBlock is appended
        # or False if it is not.
        ret = {}
        cdef cpp_map[string, int] to_add
        cdef cpp_map[string, int].iterator it
        cdef size_t k
        for k in range(self._name_func.size()):
            to_add = self._name_func[k](self, append)
            it = to_add.begin()
            while it != to_add.end():
                ret[str_cast(deref(it).first)] = deref(it).second
                inc(it)
        return ret

    cdef object _add_readability(self):
        if self.do_readability:
            ret = {
                'ancestors': self.ancestors_write,
                'readability_class_weights': self.class_weights
            }
            self.class_weights.clear()
            return ret
        else:
            return {}

    cdef void add_block_to_results(self, list results):
        """Create a block from the current partial block
        and append it to results.  Reset the partial block"""

        # compute block and link tokens!
        block_tokens = _tokens_from_text(self.text, not self.do_all_whitespace)
        cdef size_t k
        cdef string cssa
        if len(block_tokens) > 0:
            # only process blocks with something other then white space
            block_text = b' '.join(block_tokens)
            link_text = b' '.join(self.link_tokens)

            # compute link/text density
            at = re_tokenizer_nounicode.split(link_text)
            bt = re_tokenizer_nounicode.split(block_text)
            link_d = float(len(at)) / len(bt)

            lines = int(math.ceil(len(block_text) / 80.0))
            if lines == 1:
                text_d = float(len(bt))
            else:
                # need the number of tokens excluding the last partial line
                tokens = re_tokenizer_nounicode.split(
                    block_text[:(int(lines - 1) * 80)])
                text_d = len(tokens) / (lines - 1.0)

            # get the id, class attributes
            css = {}
            if self.do_css:
                for k in range(self.css_attrib.size()):
                    cssa = self.css_attrib[k]
                    css[cssa] = b' '.join(
                        _tokens_from_text(self.css[cssa], True)).lower()

            kwargs = self._add_readability()
            kwargs.update(self._extract_features(True))
            kwargs['block_start_tag'] = self.block_start_tag
            kwargs['block_start_element'] = self.block_start_element
            results.append(Block(block_text, link_d, text_d, self.anchors,
                                 self.link_tokens, css, **kwargs))
        else:
            self._extract_features(False)

        self.reinit()
        if self.do_css:
            self.reinit_css(False)

    cdef void add_text(self, cetree.tree.xmlNode *ele, string text_or_tail):
        """Add the text/tail from the element
        text_or_tail is 'text' or 'tail'"""
        cdef object t
        try:
            if text_or_tail == CTEXT:
                t = cetree.textOf(ele)
            else:
                t = cetree.tailOf(ele)
            if t is not None:
                self.text.push_back(t.encode('utf-8'))
        except UnicodeDecodeError:
            pass


    cdef void add_anchor(self, cetree.tree.xmlNode* ele, cetree._Document doc):
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
        anchor_tokens = _tokens_from_text(anchor_text_list, not self.do_all_whitespace)
        for k in range(len(anchor_tokens)):
            self.link_tokens.push_back(anchor_tokens[k])


    cdef void _tag_fe(self, string tag):
        # call the tag_featurename functions
        cdef size_t k
        for k in range(self._tag_func.size()):
            self._tag_func[k](self, tag)

    cdef void _subtree_fe(self, int start_or_end):
        # call the subtree_featurename functions
        cdef size_t k
        for k in range(self._subtree_func.size()):
            self._subtree_func[k](self, start_or_end)

    cdef void subtree_readability(self, int start_or_end):
        if start_or_end == 1:
            self.ancestors.push_back(self.tag_id)
        else:
            self.tag_id = self.ancestors.back()
            self.ancestors.pop_back()

    cdef void readability_score_node(self, cetree.tree.xmlNode* node):
        cdef int weight = 0
        cdef cetree.tree.xmlAttr* attr
        cdef size_t k
        id_class = ''
        cdef string attrib
        cdef string tag

        # check to see if we've already scored this tag_id.
        # if so don't score again
        if (self.class_weights_written.find(self.tag_id) !=
                self.class_weights_written.end()):
            return

        # first the class/id weights
        for k in range(self.css_attrib.size()):
            attrib = self.css_attrib[k]
            attr = cetree.tree.xmlHasProp(node,
                <cetree.tree.const_xmlChar*> attrib.c_str())
            if attr is not NULL:
                id_class = cetree.attributeValue(node, attr)
                if re_readability_negative.search(id_class):
                    weight -= 25
                if re_readability_positive.search(id_class):
                    weight += 25

        # now the tag name specific weight
        tag = <string> cetree.namespacedName(node).encode('utf-8')
        if tag == DIV:
            weight += 5
        elif READABILITY_PLUS3.find(tag) != READABILITY_PLUS3.end():
            weight += 5
        elif READABILITY_MINUS3.find(tag) != READABILITY_MINUS3.end():
            weight -= 3
        elif READABILITY_MINUS5.find(tag) != READABILITY_MINUS5.end():
            weight -= 5

        # finally store it
        self.class_weights.push_back(pair[uint32_t, int](self.tag_id, weight))
        self.class_weights_written.insert(self.tag_id)

    cdef void reinit_readability(self):
        self.ancestors_write = self.ancestors

    cdef void recurse(self, cetree.tree.xmlNode* subtree, list results,
        cetree._Document doc):

        cdef cetree.tree.xmlNode *node, *next_node
        cdef string tag

        # for CSS, we want to output all CSS tags for all levels in subtree
        # we will add them on entry, and pop them on exit
        if self.do_css:
            self.update_css(subtree, True)

        self._subtree_fe(1)
        if self.do_readability:
            self.readability_score_node(subtree)

        # first iteration through need to set
        if self.block_start_element is None:
            self.block_start_element = cetree.elementFactory(doc, subtree)

        # iterate through children
        if cetree.hasChild(subtree):
            node = cetree.findChild(subtree, 0)
            self.tag_id = self.next_tag_id
            self.next_tag_id += 1
        else:
            node = NULL

        while node != NULL:

            # readability
            # update the tag_id.  we do it here so it's updated for every
            # potential parent
            self.tag_id = self.next_tag_id
            self.next_tag_id += 1

            # get the tag
            tag = <string> cetree.namespacedName(node).encode('utf-8')

            if self._tag_func.size() > 0:
                self._tag_fe(tag)

            if IGNORELIST.find(tag) != IGNORELIST.end():
                # in the blacklist
                # in this case, skip the entire tag,
                # but it might have some tail text we need
                self.add_text(node, CTAIL)

            elif BLOCKS.find(tag) != BLOCKS.end():
                # this is the start of a new block
                # add the existing block to the list,
                # start the new block and recurse
                self.add_block_to_results(results)
                self.block_start_tag = tag
                self.block_start_element = cetree.elementFactory(doc, node)
                self.add_text(node, CTEXT)
                if self.do_css:
                    self.update_css(node, False)
                self.recurse(node, results, doc)
                self.add_text(node, CTAIL)

            elif tag == A:
                # an anchor tag
                self.add_anchor(node, doc)
                if self.do_css:
                    self.update_css(node, False)

            elif tag == BR and self.do_all_whitespace:
                # Add a new line to text
                self.text.push_back(NEWLINE)

            else:
                # a standard tag.
                # we need to get its text and then recurse over the subtree
                self.add_text(node, CTEXT)
                if self.do_css:
                    self.update_css(node, False)
                self.recurse(node, results, doc)
                self.add_text(node, CTAIL)

            # reset for next iteration
            next_node = cetree.nextElement(node)
            node = next_node

        if self.do_css:
            self.pop_css_tree()
        self._subtree_fe(-1)

    cdef void update_css(self, cetree.tree.xmlNode *child, bool tree):
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
            if attr is not NULL:
                deref(css_to_update)[attrib].push_back(
                    <string>cetree.attributeValue(child, attr).encode('utf-8'))


    cdef void pop_css_tree(self):
        """pop the last entry off the css lists"""
        cdef size_t k
        for k in range(self.css_attrib.size()):
            if self.css_tree[self.css_attrib[k]].size() > 0:
                self.css_tree[self.css_attrib[k]].pop_back()


cdef class TagCountPB(PartialBlock):
    """Counts tags to compute content-tag ratios"""

    # Associate with each block a tag count = the count of tags
    #   in the block.
    # Since we don't output empty blocks, we also keep track of the
    # tag count since the last block we output as an additional feature
    #

    # _tc = tag count in the current block, since the last <div>, <p>, etc.
    # _tc_lb = tag count since last block.  This is the tag count in prior
    # empty blocks, accumulated since the last block was output, excluding
    # the current block

    # so tc gets updated with each tag
    # tc is reset on block formation, even for empty blocks
    #
    # tc_lb is reset to 0 on block output
    # tc_lb accumulates tc on empty block formation
    #

    cdef int _tc, _ac, _tc_lb, _current_depth
    cdef int _min_depth_last_block, _min_depth_last_block_pending

    def __init__(self, *args, **kwargs):
        PartialBlock.__init__(self, *args, **kwargs)

        # don't need to add reinit since it's empty
        #self._reinit_func.push_back(<reinit_t>TagCountPB.reinit_tagcount)
        self._subtree_func.push_back(<subtree_t>TagCountPB.subtree_tagcount)
        self._name_func.push_back(<name_t>TagCountPB.tagcount)
        self._tag_func.push_back(<callback_t>TagCountPB.tag_tagcount)

        # will keep track of tag count and tag count since last block
        self._tc = 1  # for the top level HTML tag
        self._ac = 0  # anchor count
        self._tc_lb = 0
        self._current_depth = 0
        self._min_depth_last_block = 0
        self._min_depth_last_block_pending = 0

    cdef void reinit_tagcount(self):
        pass

    cdef void subtree_tagcount(self, int start_or_end):
        self._current_depth += start_or_end
        self._min_depth_last_block_pending = int_min(
            self._min_depth_last_block_pending, self._current_depth)

    cdef cpp_map[string, int] tagcount(self, bool append):
        # here we assume that tc is updated
        # before features are extracted
        # that is, _tc includes the current tag
        # so we adjust by -1 on output
        #
        # since tc has already been updated
        cdef cpp_map[string, int] ret
        ret.clear()
        if append:
            ret[TAGCOUNT_SINCE_LAST_BLOCK] = self._tc_lb
            ret[TAGCOUNT] = self._tc - 1
            ret[ANCHOR_COUNT] = self._ac
            ret[MIN_DEPTH_SINCE_LAST_BLOCK] = self._min_depth_last_block
            self._tc_lb = 0
            self._tc = 1
            self._ac = 0
            self._min_depth_last_block_pending = self._current_depth
            self._min_depth_last_block = self._current_depth
        else:
            self._tc_lb += (self._tc - 1)
            self._tc = 1
            self._ac = 0
        return ret

    cdef void tag_tagcount(self, string tag):
        self._tc += 1

        if tag == A:
            self._ac += 1

        if BLOCKS.find(tag) == BLOCKS.end():
            self._min_depth_last_block = self._min_depth_last_block_pending


def guess_encoding(markup, default='utf-8'):
    """
    Try to guess the encoding of ``markup`` by checking the XML declaration
    and the HTML meta tag.

    if default=CHARDET then use chardet to guess the default
    """
    xml_endpos = 1024
    html_endpos = max(2048, int(len(markup) * 0.05))
    mo = RE_XML_ENCODING.search(markup, endpos=xml_endpos)
    if mo:
        return mo.group(1)
    moh = RE_HTML_ENCODING.search(markup, endpos=html_endpos)
    if moh:
        return moh.group(1)
    if default.lower() == 'chardet':
        import chardet
        return chardet.detect(markup)['encoding']
    return default


class Blockifier(object):
    """
    A blockifier for web-page de-chroming that loosely follows the approach in
    Kohlschütter et al.: http://www.l3s.de/~kohlschuetter/publications/wsdm187-kohlschuetter.pdf

    Implements the blockify interface.
    """

    @staticmethod
    def blocks_from_tree(tree, pb=PartialBlock, do_css=True, do_readability=False, do_all_whitespace=False):
        cdef list results = []
        cdef cetree._Element ctree

        cdef PartialBlock partial_block = pb(do_css, do_readability, do_all_whitespace)
        ctree = tree
        partial_block.recurse(ctree._c_node, results, ctree._doc)

        # make the final block
        partial_block.add_block_to_results(results)

        return results

    @staticmethod
    def blockify(s, encoding=None,
                 pb=PartialBlock, do_css=True, do_readability=False,
                 parse_callback=None, do_all_whitespace=False):
        """
        Given HTML string ``s`` return a sequence of blocks with text content.

        Args:
            s (str): HTML document as a string
            encoding (str): encoding of ``s``; if None (encoding unknown), the
                original encoding will be guessed from the HTML itself
            pb
            do_css (bool): if True, add CSS-related attributes to blocks
            do_readability (bool): if True, add readability-related attributes
                to blocks
            parse_callback (Callable): if not None, will be called on the
                result of parsing in order to modify state for [reasons]

        Returns:
            List[Block]: ordered sequence of blocks with text content
        """
        # First, we need to parse the thing
        s = bytes_cast(s) # ensure we're working w/ bytes
        encoding = encoding or guess_encoding(s, default='utf-8')
        try:
            html = etree.fromstring(s,
                etree.HTMLParser(recover=True, encoding=encoding,
                remove_comments=True, remove_pis=True))
        except:
            raise BlockifyError, 'Could not blockify HTML'
        if html is None:
            # lxml sometimes doesn't raise an error but returns None
            raise BlockifyError, 'Could not blockify HTML'

        blocks = Blockifier.blocks_from_tree(html, pb, do_css, do_readability, do_all_whitespace)

        if parse_callback is not None:
            parse_callback(html)

        if do_all_whitespace:
            return str_block_list_cast(blocks)
        else:
            # only return blocks with some text content
            return [ele for ele in str_block_list_cast(blocks) if RE_TEXT.search(ele.text)]


class TagCountBlockifier(Blockifier):
    @staticmethod
    def blockify(s, encoding=None, parse_callback=None):
        return Blockifier.blockify(s, encoding=encoding, pb=TagCountPB,
                                   do_css=True, do_readability=False,
                                   parse_callback=parse_callback)

class TagCountNoCSSBlockifier(Blockifier):
    @staticmethod
    def blockify(s, encoding=None, parse_callback=None):
        return Blockifier.blockify(s, encoding=encoding, pb=TagCountPB,
                                   do_css=False, do_readability=False,
                                   parse_callback=parse_callback)

class TagCountReadabilityBlockifier(Blockifier):
    @staticmethod
    def blockify(s, encoding=None, parse_callback=None):
        return Blockifier.blockify(s, encoding=encoding, pb=TagCountPB,
                                   do_css=True, do_readability=True,
                                   parse_callback=parse_callback)

class TagCountNoCSSReadabilityBlockifier(Blockifier):
    @staticmethod
    def blockify(s, encoding=None, parse_callback=None):
        return Blockifier.blockify(s, encoding=encoding, pb=TagCountPB,
                                   do_css=False, do_readability=True,
                                   parse_callback=parse_callback)

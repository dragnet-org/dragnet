#! /usr/bin/env python
# -*- coding: utf-8 -*-

# A /rough/ implementation of that described by Kohlschütter et al.:
#    http://www.l3s.de/~kohlschuetter/publications/wsdm187-kohlschuetter.pdf

import re
from lxml import etree
import numpy as np
from itertools import chain
import scipy.weave

class Block(object):
    def __init__(self, text, link_density, text_density, anchors, link_tokens):
        self.text = text
        self.link_density = link_density
        self.text_density = text_density
        self.anchors = anchors
        self.link_tokens = link_tokens  # a hook for testing

class BlockifyError(Exception):
    """Raised when there is a fatal problem in blockify
    (if lxml fails to parse the document)
    """
    pass


class PartialBlock(object):
    """As we create blocks by recursing through subtrees
    in KohlschuetterBase, we need to maintain some state
    of the incomplete blocks.

    This class maintains that state, as well as provides methods
    to modify it."""

    def __init__(self):
        self.reinit()

    def reinit(self):
        self.text = []
        self.link_tokens = []
        self.anchors = []


    def add_block_to_results(self, results):
        """Create a block from the current partial block
        and append it to results.  Reset the partial block"""
        # compute block and link tokens!
        block_tokens = PartialBlock.tokens_from_text(self.text)
        if len(block_tokens) > 0:
            # only process blocks with something other then white space
            block_text = ' '.join(block_tokens)
            link_text = ' '.join(PartialBlock.tokens_from_text(self.link_tokens))

            # compute link/text density
            link_d = PartialBlock.link_density(block_text, link_text)
            text_d = PartialBlock.text_density(block_text)

            results.append(Block(block_text, link_d, text_d, self.anchors, self.link_tokens))
        self.reinit()

    def add_text(self, ele, text_or_tail):
        """Add the text/tail from the element
        element is an etree element
        text_or_tail is 'text' or 'tail'"""
        # we need to wrap it in a try/catch block
        # if there is an invalid unicode byte in the HTML,
        # lxml raises an error when we try to get the .text or .tail
        try:
            text = getattr(ele, text_or_tail)
            if text is not None:
                self.text.append(text)
        except UnicodeDecodeError:
            pass

    def add_anchor(self, tree):
        """Add the anchor tag to the block.
        tree = the etree a element"""
        self.anchors.append(tree)
        # need all the text from the subtree
        anchor_text_list = PartialBlock._text_from_subtree(tree, tags_exclude=KohlschuetterBase.blacklist, tail=False)
        self.text.extend(anchor_text_list)
        try:
            self.text.append(tree.tail or '')
        except UnicodeDecodeError:
            pass
        self.link_tokens.extend(PartialBlock.tokens_from_text(anchor_text_list))


    @staticmethod
    def _text_from_subtree(tree, tags_exclude=set(), tail=True):
        """Get all the text
        from the subtree, excluding tags_exclude
        If tail=False, then don't append the tail for this top level element"""
        try:
            text = [tree.text or '']
        except UnicodeDecodeError:
            text = []
        for child in tree.iterchildren():
            if child.tag not in tags_exclude:
                text.extend(PartialBlock._text_from_subtree(child, tags_exclude=tags_exclude))
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


    @staticmethod
    def tokens_from_text(text):
        """given a of text (as built in self.text for example)
        return a list of tokens.

        according to 
        http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
        chain is the fastest way to combine the list of lists"""
        return list(chain.from_iterable([re.split('\s+', ele.strip()) for ele in text if ele.strip() != '']))

    @staticmethod
    def link_density(block_text, link_text):
        '''
        Assuming both input texts are stripped of excess whitespace, return the 
        link density of this block
        '''
        # NOTE: in the case where link_text == '', this re.split
        # returns [''], which incorrectly
        # has 1 token by it's length instead of 0
        # however, fixing this bug decreases model performance by about 1%,
        # so we keep it
        anchor_tokens = re.split(r'\W+', link_text)
        block_tokens  = re.split(r'\W+', block_text)
        return float(len(anchor_tokens)) / len(block_tokens)


    @staticmethod
    def text_density(block_text):
        '''
        Assuming the text has been stripped of excess whitespace, return text
        density of this block
        '''
        import math
        lines  = math.ceil(len(block_text) / 80.0)

        if int(lines) == 1:
            tokens = re.split(r'\W+', block_text)
            return float(len(tokens))
        else:
            # need the number of tokens excluding the last partial line
            tokens = re.split(r'\W+', block_text[:(int(lines - 1) * 80)])
            return len(tokens) / (lines - 1.0)


    @staticmethod
    def recurse(subtree, partial_block, results):
        # both partial_block and results are modified
        for child in subtree.iterchildren():

            if child.tag in KohlschuetterBase.blacklist:
                # in this case, skip the entire tag,
                # but it might have some tail text we need
                partial_block.add_text(child, 'tail')
                continue

            elif child.tag in KohlschuetterBase.blocks:
                # this is the start of a new block
                # add the existing block to the list,
                # start the new block and recurse
                partial_block.add_block_to_results(results)
                partial_block.add_text(child, 'text')
                PartialBlock.recurse(child, partial_block, results)
                partial_block.add_text(child, 'tail')

            elif child.tag == 'a':
                # an anchor tag
                partial_block.add_anchor(child)

            else:
                # a standard tag.
                # we need to get it's text and then recurse over the subtree
                partial_block.add_text(child, 'text')
                PartialBlock.recurse(child, partial_block, results)
                partial_block.add_text(child, 'tail')


class KohlschuetterBase(object):
    """A base class for web-page de-chroming that loosely follows the approach in
        Kohlschütter et al.:
        http://www.l3s.de/~kohlschuetter/publications/wsdm187-kohlschuetter.pdf
      In this approach a machine learning model is used to identify blocks of text
      as content or not.

      This base class contains functionality to blockify an input HTML page.
      Subclasses implement the feature extraction and machine learning model
      for a particular approach via the methods
    """

    # All of these tags will be /completely/ ignored
    blacklist = set([
        etree.Comment, 'applet', 'area', 'base', 'basefont', 'bdo', 'button', 
        'caption', 'dfn', 'dir', 'fieldset', 'form', 'fram', 'frameset', 
        'iframe', 'img', 'input', 'legend', 'link', 'map', 'menu', 'meta', 
        'noframes', 'noscript', 'object', 'optgroup', 'option', 'param', 
        'script', 'select', 'style', 'textarea', 'var', 'xmp',

        'like', 'like-box', 'plusone', 'address',

        'code', 'pre'
    ])
    
    # Only these should be considered as housing blocks of text
    blocks = set([
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'table'
    ])

    re_non_alpha = re.compile('[\W_]', re.UNICODE)

    @staticmethod
    def blocks_from_tree(tree):
        results = []
        partial_block = PartialBlock()

        PartialBlock.recurse(tree, partial_block, results)

        # make the final block
        partial_block.add_block_to_results(results)
        return results
    

    @staticmethod
    def blockify(s):
        '''
        Take a string of HTML and return a series of blocks
        '''
        # First, we need to parse the thing
        try:
            html = etree.fromstring(s, etree.HTMLParser(recover=True))
        except:
            raise BlockifyError
        if html is None:
            # lxml sometimes doesn't raise an error but returns None
            raise BlockifyError

        blocks = KohlschuetterBase.blocks_from_tree(html)
        # only return blocks with some text content
        return [ele for ele in blocks if KohlschuetterBase.re_non_alpha.sub('', ele.text) != '']


    def analyze(self, s, blocks=False):
        """s = HTML string
        returns the content as a string, or if `block`, then the blocks
        themselves are returned.
        """
        features, blocks_ = self.make_features(s)
        if features is not None:
            content_mask = self.block_analyze(features)
            results = [ele[0] for ele in zip(blocks_, content_mask) if ele[1]]
        else:
            # doc is too short. return all content
            results = blocks_
        if blocks:
            return results
        return ' '.join(blk.text for blk in results)


    @staticmethod
    def plot(blocks, content_mask):
        import pylab as plt

        fig = plt.figure(1)
        fig.clf()
        block_lengths = np.array([len(ele.text) for ele in blocks]) - 1.0
        block_lengths_content = block_lengths.copy()
        block_lengths_content[~np.array(content_mask)] = 0.0
        block_lengths_no_content = block_lengths.copy()
        block_lengths_no_content[content_mask] = 0.0

        ret = plt.bar(np.arange(len(blocks)), block_lengths_no_content, 0.5)
        ret = plt.bar(np.arange(len(blocks)), block_lengths_content, 0.5)

        fig.show()




class Kohlschuetter(KohlschuetterBase):
    @staticmethod
    def make_features(s):
        """s = the HTML string
        return a numpy array of the features + the associated raw content"""
        blocks = KohlschuetterBase.blockify(s)

        # doc needs to be at least three blocks, otherwise return everything
        if len(blocks) < 3:
            return None, blocks

        features = np.zeros((len(blocks), 6))
        for i in range(1, len(blocks)-1):
            previous = blocks[i-1]
            current  = blocks[i]
            next     = blocks[i+1]
            features[i, :] = [previous.link_density, previous.text_density,
                                  current.link_density, current.text_density,
                                  next.link_density, next.text_density]
        i = 0
        features[0, :] = [0.0, 0.0,
                          blocks[i].link_density, blocks[i].text_density,
                          blocks[i + 1].link_density, blocks[i + 1].text_density]
        i = len(blocks) - 1
        features[-1, :] = [blocks[i - 1].link_density, blocks[i - 1].text_density,
                          blocks[i].link_density, blocks[i].text_density,
                          0.0, 0.0]
        return features, blocks

    def block_analyze(self, features):
        """Takes a the features
        Returns a list True/False of ones classified as content
        
        Note: this is the decision tree published in the original paper
        We benchmarked it against our data set and it performed poorly,
        and we attribute it to differences the blockify implementation.
        """
        # curr_linkDensity <= 0.333333
        # | prev_linkDensity <= 0.555556
        # | | curr_textDensity <= 9
        # | | | next_textDensity <= 10
        # | | | | prev_textDensity <= 4: BOILERPLATE
        # | | | | prev_textDensity > 4: CONTENT
        # | | | next_textDensity > 10: CONTENT
        # | | curr_textDensity > 9
        # | | | next_textDensity = 0: BOILERPLATE
        # | | | next_textDensity > 0: CONTENT
        # | prev_linkDensity > 0.555556
        # | | next_textDensity <= 11: BOILERPLATE
        # | | next_textDensity > 11: CONTENT
        # curr_linkDensity > 0.333333: BOILERPLATE

        results = []
        
        for i in xrange(features.shape[0]):
            (previous_link_density, previous_text_density,
            current_link_density, current_text_density,
            next_link_density, next_text_density) = features[i, :]
            if current_link_density <= 0.333333:
                if previous_link_density <= 0.555556:
                    if current_text_density <= 9:
                        if next_text_density <= 10:
                            if previous_text_density <= 4:
                                # Boilerplate
                                results.append(False)
                            else: # previous.text_density > 4
                                results.append(True)
                        else: # next.text_density > 10
                            results.append(True)
                    else: # current.text_density > 9
                        if next_text_density == 0:
                            results.append(False)
                        else: # next.text_density > 0
                            results.append(True)
                else: # previous.link_density > 0.555556
                    if next_text_density <= 11:
                        # Boilerplate
                        results.append(False)
                    else: # next.text_density > 11
                        results.append(True)
            else: # current.link_density > 0.333333
                # Boilerplace
                results.append(False)

        return results

def normalize_features(features, mean_std):
    """Normalize the features IN PLACE.
       mean_std = {'mean':[list of means],
                   'std':[list of std] }
        mean_std HAS AN OPTIONAL 'log' key.
            If present, it includes a flag whether to take a log first.
            If not None, then it gives a value to do a transform:
                log(x + exp(-value)) + value
       the lists are the same length as features.shape[1]
       
       if features is None, then do nothing"""
    if features is not None:
        if 'log' in mean_std:
            for k in xrange(features.shape[1]):
                if mean_std['log'][k] is not None:
                    features[:, k] = np.log(features[:, k] + np.exp(-mean_std['log'][k])) + mean_std['log'][k]

        for k in xrange(features.shape[1]):
            features[:, k] = (features[:, k] - mean_std['mean'][k]) / mean_std['std'][k]


class KohlschuetterNormalized(Kohlschuetter):
    """Use the Kohlschuetter, but do some mean/std normalization"""

    @staticmethod
    def load_mean_std(mean_std):
        """mean_std is either a filename string,
        or a json blob.
        if a string, load it, otherwise just return"""
        import json
        if isinstance(mean_std, basestring):
            return json.load(open(mean_std, 'r'))
        else:
            return mean_std

    def __init__(self, mean_std):
        """mean_std = the json file with mean/std of the features
        mean_std = {'mean':[list of means],
                    'std':[list of std] """
        Kohlschuetter.__init__(self)
        self._mean_std = KohlschuetterNormalized.load_mean_std(mean_std)

    def make_features(self, s):
        "Make text, anchor features and some normalization"
        features, blocks = Kohlschuetter.make_features(s)
        normalize_features(features, self._mean_std)
        return features, blocks


re_capital = re.compile('[A-Z]')
re_digit = re.compile('\d')
def capital_digit_features(blocks):
    """percent of block that is capitalized and numeric"""
    features = np.zeros((len(blocks), 2))
    features[:, 0] = [len(re_capital.findall(ele.text)) / float(len(ele.text)) for ele in blocks]
    features[:, 1] = [len(re_digit.findall(ele.text)) / float(len(ele.text)) for ele in blocks]
    return features


def token_feature(blocks):
    """A global token count feature"""
    from collections import defaultdict
    word_dict = defaultdict(lambda: 0)
    block_tokens = []
    token_count = 0
    for block in blocks:
        block_tokens.append(re.split('\W', block.text.strip()))
        for token in block_tokens[-1]:
            word_dict[token] += 1
            token_count += 1

    token_count = float(token_count)

    nblocks = len(blocks)
    feature = np.zeros(nblocks)
    for k in xrange(nblocks):
        ntokens = len(block_tokens[k])

        for token in block_tokens[k]:
            feature[k] += np.log(word_dict[token])

        feature[k] = feature[k] / ntokens - np.log(token_count)

        if np.isinf(feature[k]):
            feature[k] = -10.0   # just in case

    return feature



class KohlschuetterExpanded(KohlschuetterNormalized):
    """An model that takes the Kohlschuetter features and adds
    some additional ones"""
    def make_features(self, s):
        from scipy import percentile

        features_koh, blocks = Kohlschuetter.make_features(s)
        if features_koh is None:
            return None, blocks

        features = np.zeros((features_koh.shape[0], 6 + 4 + 2))
        features[:, :6] = features_koh[:]

        # a global feature based on connected blocks of long text
        # inspired by Arias
        block_lengths = np.array([len(block.text) for block in blocks])
        index = block_lengths.argmax()
        k = 6
        for c in [0.15, 0.3333]:
            for window in [1, 4]:
                cutoff = int(percentile(block_lengths, 97) * c)
                lowindex, highindex = KohlschuetterExpanded.strip(block_lengths, index, window, cutoff)
                features[lowindex:(highindex + 1), k] = 1.0
                k += 1

        features[:, -2:] = capital_digit_features(blocks)
        normalize_features(features, self._mean_std)
        return features, blocks


    @staticmethod
    def strip(block_lengths, index, window, cutoff):
        ret = np.zeros(2, np.int)
        nblock = len(block_lengths)
        c_code = """
            // First we'll work backwards to find the beginning index, and then we'll
            // work forward to find the ending index, and then we'll just take that
            // slice to be our content
            int lowindex  = index;
            int lastindex = index;
            while (lowindex > 0)
            {
                if (lastindex - lowindex > window)
                    break;
                if (block_lengths(lowindex) >= cutoff)
                    lastindex = lowindex;
                lowindex--;
            }
            ret(0) = lastindex;

            // Like above, except we're looking in the forward direction
            int highindex = index;
            lastindex = index;
            while (highindex < nblock)
            {
                if (highindex - lastindex > window)
                    break;
                if (block_lengths(highindex) >= cutoff)
                    lastindex = highindex;
                highindex++;
            }
            ret(1) = lastindex;
        """
        scipy.weave.inline(c_code,
                ['ret', 'nblock', 'index', 'window', 'cutoff', 'block_lengths'],
                type_converters=scipy.weave.converters.blitz)
        return ret


class DragnetModel(KohlschuetterBase):
    """
    Machine learning models that predict whether
    blocks are content or not
    """
    def __init__(self, block_model, threshold=0.5): 
        """block_model is a model with these methods:
                block_model.train(X, y) = train the model
                block_model.pred(X) = make predictions (0-1)
           threshold = anything with block_model.pred(X) > threshold
                is considered content
        """
        KohlschuetterBase.__init__(self)
        self.block_model = block_model
        self._threshold = threshold

    def block_analyze(self, features):
        return self.block_model.pred(features) > self._threshold

class DragnetModelKohlschuetterFeatures(DragnetModel, KohlschuetterNormalized):
    """machine learning models that use the two features from
    Kohlschuetter as the model features"""
    # according to use Python's MRO DragnetModelBase.block_analyze
    # is inherited 
    def __init__(self, block_model, mean_std, threshold=0.5):
        DragnetModel.__init__(self, block_model, threshold)
        KohlschuetterNormalized.__init__(self, mean_std)


class DragnetModelKohlschuetterExpanded(DragnetModel, KohlschuetterExpanded):
    """machine learning models that use an expanded set of features"""
    # according to use Python's MRO DragnetModelBase.block_analyze
    # is inherited 
    def __init__(self, block_model, mean_std, threshold=0.5):
        DragnetModel.__init__(self, block_model, threshold)
        KohlschuetterExpanded.__init__(self, mean_std)







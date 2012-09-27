#! /usr/bin/env python
# -*- coding: utf-8 -*-

# A /rough/ implementation of that described by Kohlschütter et al.:
#    http://www.l3s.de/~kohlschuetter/publications/wsdm187-kohlschuetter.pdf

import re
from lxml import etree
import numpy as np

class Block(object):
    def __init__(self, text, link_density, text_density):
        self.text         = re.sub(r'\s+', ' ', text or '')
        self.link_density = link_density
        self.text_density = text_density

class KohlschuetterBase(object):
    """A base class for web-page de-chroming that loosely follows the approach in
        Kohlschütter et al.:
        http://www.l3s.de/~kohlschuetter/publications/wsdm187-kohlschuetter.pdf
      In this approach a machine learning model is used to identify blocks of text
      as content or not.

      This base class contains functionality to blockify an input HTML page.
      Subclasses implement the feature extraction and machine learning model
      for a particular approach via the methods
           self.make_features(s) where s = the HTML string
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
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div'
    ])
    
    # Tags to treat as contiguous text
    ignore = set([
        'a', 'abbr', 'acronym', 'big', 'cide', 'del', 'em', 'font', 'i', 'ins', 
        'q', 's', 'small', 'span', 'strike', 'strong', 'sub', 'sup', 'u', 
        'noscript'
    ])

    # Tags that manipulate the structure
    structure = set([
        'blockquote', 'code', 'dd', 'dfn', 'dir', 'div', 'dl', 'dt', 'h1', 'h2', 
        'h3', 'h4', 'h5', 'h6', 'kbd', 'li', 'ol', 'p', 'pre', 'samp', 'table', 
        'tbody', 'td', 'th', 'thead', 'tr', 'tt', 'ul', 'xmp'
    ])


    @staticmethod
    def link_density(block_text, link_text):
        '''
        Assuming both input texts are stripped of excess whitespace, return the 
        link density of this block
        '''
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
        block_text = re.sub(r'\s+', ' ', block_text or '')
        lines  = math.ceil(len(block_text) / 80.0)
        
        if int(lines) == 1:
            tokens = re.split(r'\W+', block_text)
            return float(len(tokens))
        else:
            # need the number of tokens excluding the last partial line
            tokens = re.split(r'\W+', block_text[:(int(lines - 1) * 80)])
            return len(tokens) / (lines - 1.0)
    
    @staticmethod
    def text(tree):
        '''Recursively get text for a sub-tree'''
        def strip_whitespace(t, what):
            # sometimes we get a tree.text that we can't encode or decode
            # even with ignoring errors? it raises a UnicodeDecodeError with
            # print(tree.text) and type(tree.text) ?
            # and even on assignment tt = tree.text ???
            # error is raised from lxml so maybe this is an lxml bug?
            try:
                if what == 'text':
                    ret = re.sub(r'\s+', ' ', t.text or '')
                else:  # what == 'tail':
                    ret = re.sub(r'\s+', ' ', t.tail or '')
            except UnicodeDecodeError:
                ret = ''
            return ret

        text = strip_whitespace(tree, 'text')
        for child in tree:
            text += ' ' + KohlschuetterBase.text(child)
            text += strip_whitespace(tree, 'child')
        return text
    
    @staticmethod
    def recurse(tree):
        """
            tree = element tree HTML parse tree
        """
        results = []
        text      = tree.text or ''
        link_text = ''
        
        for child in tree:
            if child.tag in KohlschuetterBase.blacklist:
                continue
            elif child.tag == 'a':
                # It's an anchor! Grow it out
                t = KohlschuetterBase.text(child)
                text += ' ' + t
                link_text += ' ' + t
            elif child.tag in KohlschuetterBase.ignore:
                # It's just something to glom together
                text += ' ' + KohlschuetterBase.text(child)
            else:
                # This is a new block; append the current block to results
                if text and tree.tag in KohlschuetterBase.blocks:
                    link_d = KohlschuetterBase.link_density(text, link_text)
                    text_d = KohlschuetterBase.text_density(text)
                    results.append(Block(text, link_d, text_d))
                
                results.extend(KohlschuetterBase.recurse(child))

                # Reset the text, link_text
                text      = re.sub(r'\s+', ' ', child.tail or '')
                link_text = ''
            
            # Now append the tail
            text += ' ' + re.sub(r'\s+', ' ', child.tail or '')
        
        if text and tree.tag in KohlschuetterBase.blocks:
            link_d = KohlschuetterBase.link_density(text, link_text)
            text_d = KohlschuetterBase.text_density(text)
            results.append(Block(text, link_d, text_d))
        
        return results
    
    @staticmethod
    def blockify(s):
        '''
        Take a string of HTML and return a series of blocks
        '''
        # First, we need to parse the thing
        html = etree.fromstring(s, etree.HTMLParser(recover=True))
        blocks = KohlschuetterBase.recurse(html)
        # only return blocks with some text content
        return [ele for ele in blocks if re.sub('[\W_]', '', ele.text).strip() != '']


    def analyze(self, s, blocks=False):
        """s = HTML string
        returns the content as a string, or if `block`, then the blocks
        themselves are returned.
        """
        features, blocks_ = self.make_features(s)
        content_mask = self.block_analyze(features)
        results = [ele[0] for ele in zip(blocks_, content_mask) if ele[1]]
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
        Returns a list True/False of ones classified as content"""
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

class KohlschuetterNormalized(Kohlschuetter):
    def __init__(self, mean_std):
        """mean_std = the json file with mean/std of the features
        mean_std = {'mean':[list of means],
                    'std':[list of std] """
        import json
        Kohlschuetter.__init__(self)
        self._mean_std = json.load(open(mean_std, 'r'))

    def make_features(self, s):
        "Make text, anchor features and some normalization"
        features, blocks = Kohlschuetter.make_features(s)
        for k in xrange(features.shape[1]):
            features[:, k] = (features[:, k] - self._mean_std['mean'][k]) / self._mean_std['std'][k]
        return features, blocks


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



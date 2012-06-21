#! /usr/bin/env python
# -*- coding: utf-8 -*-

# A /rough/ implementation of that described by Kohlschütter et al.:
#    http://www.l3s.de/~kohlschuetter/publications/wsdm187-kohlschuetter.pdf

import re
from lxml import etree

class Block(object):
    def __init__(self, text, link_density, text_density):
        self.text         = re.sub(r'\s+', ' ', text or '')
        self.link_density = link_density
        self.text_density = text_density

class Kohlschuetter(object):
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
        tokens = re.split(r'\W+', block_text)
        lines  = math.ceil(len(block_text) / 80.0)
        
        if int(lines) == 1:
            return float(len(tokens))
        else:
            return len(tokens) / (lines - 1.0)
    
    @staticmethod
    def text(tree):
        '''Recursively get text for tree'''
        text = re.sub(r'\s+', ' ', tree.text or '')
        for child in tree:
            text += ' ' + Kohlschuetter.text(child)
            text += ' ' + re.sub(r'\s+', ' ', child.tail or '')
        return text
    
    @staticmethod
    def recurse(tree):
        import re
        
        results = []
        text      = tree.text or ''
        link_text = ''
        
        for child in tree:
            if child.tag in Kohlschuetter.blacklist:
                continue
            elif child.tag == 'a':
                # It's an anchor! Grow it out
                t = Kohlschuetter.text(child)
                text += ' ' + t
                link_text += ' ' + t
            elif child.tag in Kohlschuetter.ignore:
                # It's just something to glom together
                text += ' ' + Kohlschuetter.text(child)
            else:
                # This is a new block; append the current block to results
                if text and tree.tag in Kohlschuetter.blocks:
                    link_d = Kohlschuetter.link_density(text, link_text)
                    text_d = Kohlschuetter.text_density(text)
                    results.append(Block(text, link_d, text_d))
                
                results.extend(Kohlschuetter.recurse(child))
                # Reset the text, link_text
                text      = re.sub(r'\s+', ' ', child.tail or '')
                link_text = ''
            
            # Now append the tail
            text += ' ' + re.sub(r'\s+', ' ', child.tail or '')
        
        if text and tree.tag in Kohlschuetter.blocks:
            link_d = Kohlschuetter.link_density(text, link_text)
            text_d = Kohlschuetter.text_density(text)
            results.append(Block(text, link_d, text_d))
        
        return results
    
    @staticmethod
    def blockify(s):
        '''
        Take a string of HTML and return a series of blocks
        '''
        # First, we need to parse the thing
        html = etree.fromstring(s, etree.HTMLParser(recover=True))
        return Kohlschuetter.recurse(html)
      
    @staticmethod
    def analyze(s, url=''):
        blocks  = Kohlschuetter.blockify(s)
        results = []
        
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
        
        for i in range(1, len(blocks)-1):
            previous = blocks[i-1]
            current  = blocks[i]
            next     = blocks[i+1]
            if current.link_density <= 0.333333:
                if previous.link_density <= 0.555556:
                    if current.text_density <= 9:
                        if next.text_density <= 10:
                            if previous.text_density <= 4:
                                # Boilerplate
                                pass
                            else: # previous.text_density > 4
                                results.append(current.text)
                        else: # next.text_density > 10
                            results.append(current.text)
                    else: # current.text_density > 9
                        if next.text_density == 0:
                            # Boilerplate
                            pass
                        else: # next.text_density > 0
                            results.append(current.text)
                else: # previous.link_density > 0.555556
                    if next.text_density <= 11:
                        # Boilerplate
                        pass
                    else: # next.text_density > 11
                        results.append(current.text)
            else: # current.link_density > 0.333333
                # Boilerplace
                pass
        
        print len(blocks)
        return ' '.join(results).encode('utf-8')
#! /usr/bin/env python

# A /rough/ implementation of that described by Aurias et al.:
#    https://lirias.kuleuven.be/bitstream/123456789/215528/1/AriasEtAl2009.pdf
#
# There are a few differences, however. In particular, rather than basing the
# cutoff off of the maximum found string length, it bases it off of a high 
# percentile of the lengths of the input strings. A minor difference.
#
# The second major difference is that rather than using a fixed window for 
# comparing new strings for inclusion in the set, it's based on a portion of 
# the number of total strings in the list L
#
# Lastly, instead of just finding the cluster around the string with the maximum
# length, we perform this algorithm on the five longest strings in the list, and
# simply return the longest resulting string
#
# Constants and terms from the original paper are maintained. All the constants
# we introduce are prefixed with `k`, rather than `c`.

from lxml import etree

class Arias(object):
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
    
    # c1 is used as a portion of the maximum length string to determine the 
    # minimum length of a string that's considered acceptable
    c1 = 0.333
    # c2 is the most consecutive too-short strings that must be found in both 
    # directions in order to stop including strings from L. This algorithm does
    # not actually make use of this constant directly (but it appeared in the
    # original paper)
    c2 = 4
    # What percentile of the string lengths to use as the basis for calculating
    # the cutoff. In particular, cutoff = (k1'th percentile) * c1
    k1 = 0.97
    # When calculating the window size (c2), what portion of the length of the
    # list L to use 
    k2 = 0.04
    
    @staticmethod
    def recurse(tree):
        '''
        Accepts a lxml etree as input, and returns a list of strings (L) as
        described in the paper
        '''
        import re
        results = []
        
        # If the tag is considered a structural tag, then append a blank
        if tree.tag in Arias.structure:
            results.append('')
        
        # If there is any non-blank text in the tag, regardless of what type it 
        # is, append that text
        text = re.sub(r'\s+', ' ', tree.text or '')
        if len(text) > 1:
            results.append(text)
        
        # For each child...
        for child in tree:
            # If child is a tag we are explicitly not dealing with, then skip
            if child.tag in Arias.blacklist:
                continue
            
            if child.tag in Arias.ignore:
                # If it's an 'ignore' tag, we should glom its text up with the 
                # parent element's text. Or rather, the previous element's to 
                # maintain order.
                text = re.sub(r'\s+', ' ', child.text or '')
                tail = re.sub(r'\s+', ' ', child.tail or '')
                if len(results):
                    results[-1] += text + ' ' + tail
                else:
                    results.append(text)
            else:
                # If it's not an ignore tag, then recurse through this element
                results.extend(Arias.recurse(child))
        
        # Append the tail of the current element, provided it's not just space
        tail = re.sub(r'\s+', ' ', tree.tail or '')
        if len(tail) > 1:
            results.append(tail)
        
        return results
    
    @staticmethod
    def plot(L, name, low, high, cutoff):
        '''
        Helper method to plot the document (like in the paper)
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        
        # First, plot up to the low point in blue...
        p1 = plt.bar(np.arange(low), [len(l) for l in L[0:low]], linewidth=0.0)
        # And now from low-high in red
        p2 = plt.bar(np.arange(low, hi+1), [len(l) for l in L[low:hi+1]],
            linewidth=0.0, color='r')
        # And from then on in blue
        p3 = plt.bar(np.arange(hi+1, len(L)), [len(l) for l in L[hi+1:]], 
            linewidth=0.0)
        # Lastly, apply a line across the board at the cutoff
        line = plt.plot([0, len(L)], [cutoff, cutoff], 'g-')
        
        plt.xlabel('Order')
        plt.ylabel('Length')
        plt.title(name)
        plt.show()
    
    @staticmethod
    def strip(L, index, cutoff, window):
        # Given an L, an index to start examining L at, and the cutoff params, 
        # returns the low and high indexes to use for the interval
        maxL = len(L[index])
    
        # First we'll work backwards to find the beginning index, and then we'll
        # work forward to find the ending index, and then we'll just take that
        # slice to be our content
        lowindex  = index
        lastindex = index
        while lowindex:
            if lastindex - lowindex > window:
                break
            if len(L[lowindex]) >= cutoff:
                lastindex = lowindex
            lowindex -= 1
        lowindex = lastindex
        
        # Like above, except we're looking in the forward direction
        highindex = index
        lastindex = index
        while highindex < len(L):
            if highindex - lastindex > window:
                break
            if len(L[highindex]) >= cutoff:
                lastindex = highindex
            highindex += 1
        highindex = lastindex
        
        return lowindex, highindex
    
    @staticmethod
    def analyze(s, url=''):
        '''
        Given an input string (html), and an optional url (for display purposes)
        return a string that should be considered representative of the document
        '''
        # First, we need to parse the thing
        html = etree.fromstring(s, etree.HTMLParser(recover=True))
        # Transform the webpage into a sequence of strings, L
        L = Arias.recurse(html)
        
        ordered = sorted([(len(L[i]), i) for i in range(len(L))])
        
        # For a cutoff, we'll use 1/3 of the 97'th percentile
        cutoff = float(ordered[int(len(ordered) * Arias.k1)][0]) * Arias.c1
        
        # For each of the top 5, let's give it a shot, and see which one returns 
        # the longest string
        best = None
        best_low  = 0
        best_high = 0
        for l, index in ordered[-5:]:
            low, high = Arias.strip(L, index, cutoff, len(L) * Arias.k2)
            s = ' '.join(L[low:high+1]).encode('utf-8')
            if best == None:
                best      = s
                best_low  = low
                best_high = high
            elif len(best) < len(s):
                best      = s
                best_low  = low
                best_high = high
        
        return best

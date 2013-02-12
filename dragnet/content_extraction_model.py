#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np

class IdentityPredictor(object):
    """Mock out the machine learning model with an identity model."""
    @staticmethod
    def predict(x):
        return x

    @staticmethod
    def fit(*args, **kargs):
        pass

class ContentExtractionModel(object):
    """Content extraction model

    Encapsulates a blockifier, some feature generators and a
    machine learing block model
    
    Implements analyze, make_features"""

    def __init__(self, blockifier, features, block_model, threshold=0.5):
        """blockifier = implements blockify
        features = list of things that implement features interface
        block_model = sklearn interface model"""

        self._blockifier = blockifier
        self._features = features
        self._block_model = block_model
        self._threshold = threshold

        # check the features
        self._nfeatures = sum(ele.nfeatures for ele in self._features)
        for f in self._features:
            assert callable(f)

    def set_threshold(self, thres):
        """Set the threshold
        
        0<= thres <= 1.0"""
        self._threshold = thres

    def analyze(self, s, blocks=False):
        """s = HTML string
        returns the content as a string, or if `block`, then the blocks
        themselves are returned.
        """
        features, blocks_ = self.make_features(s)
        if features is not None:
            content_mask = self._block_model.predict(features) > self._threshold
            results = [ele[0] for ele in zip(blocks_, content_mask) if ele[1]]
        else:
            # doc is too short. return all content
            results = blocks_
        if blocks:
            return results
        return ' '.join(blk.text for blk in results)


    def make_features(self, s, train=False):
        """s = HTML string
           return features, blocks

           raises BlockifyError if there is an error parsing the doc
           and None if doc is too short (< 3 blocks)
           
           train = if true, then passes it into feature maker"""

        blocks = self._blockifier.blockify(s)

        # doc needs to be at least three blocks, otherwise return everything
        if len(blocks) < 3:
            return None, blocks

        # compute the features
        features = np.zeros((len(blocks), self._nfeatures))
        offset = 0
        for f in self._features:
            offset_end = offset + f.nfeatures
            features[:, offset:offset_end] = f(blocks, train)
            offset = offset_end

        return features, blocks



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


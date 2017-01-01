#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from .blocks import Blockifier


class IdentityPredictor(object):
    """Mock out the machine learning model with an identity model."""
    @staticmethod
    def predict(x):
        return x

    @staticmethod
    def fit(*args, **kargs):
        pass


class BaselinePredictor(object):
    """Always predict content"""
    @staticmethod
    def predict(x):
        return np.ones(x.shape)

    @staticmethod
    def fit(*args, **kwargs):
        pass


def nofeatures(blocks, *args, **kwargs):
    return np.zeros((len(blocks), 1))


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
        for f in self._features:
            if not callable(f):
                raise TypeError('All features must be callable')

    def set_threshold(self, thres):
        """Set the threshold

        0<= thres <= 1.0"""
        self._threshold = thres

    def analyze(self, s, blocks=False, encoding=None, parse_callback=None):
        """s = HTML string
        returns the content as a string, or if `block`, then the blocks
        themselves are returned.

        if encoding is not None, then this specifies the HTML string encoding.
            If None, then try to guess it.
        """
        # blockify html into blocks
        blocks_ = self._blockifier.blockify(
            s, encoding=encoding, parse_callback=parse_callback)

        # make features, run model and return content
        return self.analyze_from_blocks(blocks_, return_blocks=blocks)

    def analyze_from_blocks(self, blocks, return_blocks=False):
        # make features
        features = self.make_features_from_blocks(blocks)

        # run model and select results
        if features is not None:
            content_mask = self._block_model.predict(features) > self._threshold
            results = [ele[0] for ele in zip(blocks, content_mask) if ele[1]]
        else:
            # doc is too short. return all content
            results = blocks

        if return_blocks:
            return results
        else:
            return ' '.join(blk.text for blk in results)

    def make_features_from_blocks(self, blocks, train=False):
        # doc needs to be at least three blocks, otherwise return everything
        if len(blocks) < 3:
            return None
        # compute the features
        return np.column_stack(tuple(f(blocks, train) for f in self._features))

    def make_features(self, s, train=False, encoding=None, parse_callback=None):
        """s = HTML string
           return features, blocks

           raises BlockifyError if there is an error parsing the doc
           and None if doc is too short (< 3 blocks)

           train = if true, then passes it into feature maker
        """
        # note: this method is not longer needed by ContentExtractionModel
        # but is kept for now for backward compatibilty with training code
        blocks = self._blockifier.blockify(
            s, encoding=encoding, parse_callback=parse_callback)
        return self.make_features_from_blocks(blocks, train), blocks

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

        plt.bar(np.arange(len(blocks)), block_lengths_no_content, 0.5)
        plt.bar(np.arange(len(blocks)), block_lengths_content, 0.5)

        fig.show()


class ContentCommentsExtractionModel(ContentExtractionModel):
    """
    Run two models: a content only and a content + comments model
    on a document and return the output of both
    """
    def __init__(self, blockifier, features,
                 content_model, content_comments_model, threshold=0.5):
        self._blockifier = blockifier
        self._features = features
        self._content_model = content_model
        self._content_comments_model = content_comments_model
        self._threshold = threshold

        # check the features
        for f in self._features:
            if not callable(f):
                raise TypeError('All features must be callable')

    def analyze(self, s, blocks=False, encoding=None, parse_callback=None):
        """
        Get the content and content+comments

        s = HTML string
        if encoding is not None, then this specifies the HTML string encoding.
            If None, then try to guess it.
        parse_callback: if not None then this is callable that is invoked
            with the parse tree

        if blocks is False then returns a tuple of strings:
            (main_content_string, main_content_and_comments_string)
        if blocks is True then returns a tuple of block instances:
            (list of main content blocks,
             list of main content and comments blocks)
        """
        # just a wrapper around base class method for now
        # all the subclass specific logic is in analyze_from_blocks
        return super(ContentCommentsExtractionModel, self).analyze(
            s, blocks=blocks, encoding=encoding, parse_callback=parse_callback)

    def analyze_from_blocks(self, blocks_, return_blocks=False):
        features = self.make_features_from_blocks(blocks_)

        if features is not None:
            content_mask = self._content_model.predict(
                features) > self._threshold
            content_comments_mask = self._content_comments_model.predict(
                features) > self._threshold
            blocks_content = [
                ele[0] for ele in zip(blocks_, content_mask) if ele[1]]
            blocks_content_comments = [
                ele[0] for ele in zip(blocks_, content_comments_mask) if ele[1]]
        else:
            # doc is too short. return all content
            blocks_content = blocks_
            blocks_content_comments = blocks_

        if return_blocks:
            return (blocks_content, blocks_content_comments)
        else:
            return (
                ' '.join(blk.text for blk in blocks_content),
                ' '.join(blk.text for blk in blocks_content_comments)
                )


class SklearnWrapper(object):
    def __init__(self, skmodel):
        # skmodel implements predict_proba and has classes_ attribute
        self._skmodel = skmodel
        classes = list(skmodel.classes_)
        self._positive_idx = classes.index(1)

    def predict(self, X):
        return self._skmodel.predict_proba(X)[:, self._positive_idx]

baseline_model = ContentExtractionModel(Blockifier, [nofeatures], BaselinePredictor)

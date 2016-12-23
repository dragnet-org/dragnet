#! /usr/bin/env python
# -*- coding: utf-8 -*-

# A /rough/ implementation of that described by Kohlschütter et al.:
#    http://www.l3s.de/~kohlschuetter/publications/wsdm187-kohlschuetter.pdf
import numpy as np

from .content_extraction_model import ContentExtractionModel
from .blocks import Blockifier
from .compat import range_


def kohlschuetter_features(blocks, train=False):
    """The text density/link density features
    from Kohlschuetter.  Implements the features interface"""
    # need at least 3 blocks to make features
    assert len(blocks) >= 3

    features = np.zeros((len(blocks), 6))
    for i in range_(1, len(blocks) - 1):
        previous = blocks[i - 1]
        current = blocks[i]
        next_ = blocks[i + 1]
        features[i, :] = [
            previous.link_density, previous.text_density,
            current.link_density, current.text_density,
            next_.link_density, next_.text_density]
    i = 0
    features[0, :] = [0.0, 0.0,
                      blocks[i].link_density, blocks[i].text_density,
                      blocks[i + 1].link_density, blocks[i + 1].text_density]
    i = len(blocks) - 1
    features[-1, :] = [blocks[i - 1].link_density, blocks[i - 1].text_density,
                       blocks[i].link_density, blocks[i].text_density,
                       0.0, 0.0]

    return features

kohlschuetter_features.nfeatures = 6


class KohlschuetterBlockModel(object):
    """the original decision tree published in Kohlschuetter et al"""

    @staticmethod
    def predict(features):
        """Takes a the features
        Returns a list True/False of ones classified as content

        Note: this is the decision tree published in the original paper
        We benchmarked it against our data set and it performed poorly.
        We attribute this to differences the blockify implementation
        and the specific parameters in the decision tree (that will be
        highly sensitive to the details of the implementation).

        However, the Kohlschuetter features trained with a logistic
        model on the data performs quite well, so while don't use
        these parameters directly, we do adopt the overall approach.
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

        for i in range_(features.shape[0]):
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
                            else:  # previous.text_density > 4
                                results.append(True)
                        else:  # next.text_density > 10
                            results.append(True)
                    else:  # current.text_density > 9
                        if next_text_density == 0:
                            results.append(False)
                        else:  # next.text_density > 0
                            results.append(True)
                else:  # previous.link_density > 0.555556
                    if next_text_density <= 11:
                        # Boilerplate
                        results.append(False)
                    else:  # next.text_density > 11
                        results.append(True)
            else:  # current.link_density > 0.333333
                # Boilerplace
                results.append(False)

        return results

kohlschuetter = ContentExtractionModel(Blockifier, [kohlschuetter_features], KohlschuetterBlockModel)

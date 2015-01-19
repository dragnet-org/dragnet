
import numpy as np

from collections import defaultdict

from .blocks import TagCountNoCSSReadabilityBlockifier
from .content_extraction_model import ContentExtractionModel

def readability_features(blocks, *args, **kwargs):
    '''
    Features inspired by Readability
    '''
    # 1. create content_score for each tag_id
    #   read through blocks.  for each class weight written, add it's weight
    #       then: if text length > 25:
    #           add in min((text_len / 100), 3) * link density to
    #               to ancestors[-1]
    scores = defaultdict(float)
    # link density in subtrees. first entry is total text length * link density,
    # second is total text length
    ld = defaultdict(lambda: [0.0, 0.0])
    # only tag_ids that have <p> or <div> as children are valid root nodes
    valid_nodes = defaultdict(bool)

    for block in blocks:
        for tag_id, weight in block.features['readability_class_weights']:
            scores[tag_id] += weight

        if len(block.features['ancestors']) > 0:
            text_len = len(block.text)
            for ancestor in block.features['ancestors']:
                ld[ancestor][0] += block.link_density * text_len
                ld[ancestor][1] += text_len

            if text_len > 25 and (
                    block.features['block_start_tag'] in ['div', 'p']):
                parent = block.features['ancestors'][-1]
                scores[parent] += (1 + min(text_len / 100, 3))
                valid_nodes[parent] = True

    # scale scores by link density
    for tag_id in scores.iterkeys():
        scores[tag_id] *= (1.0 - ld[tag_id][0] / max(ld[tag_id][1], 1.0))

    # 2. get max score of all scores
    valid_scores = [score 
            for tag_id, score in scores.iteritems() if valid_nodes[tag_id]]
    if len(valid_scores) == 0:
        # no div, p in document!
        # return all features as 0
        return np.zeros((len(blocks), 1))

    # otherwise get the max
    max_score = max(max(valid_scores), 1.0)

    # 3. read through blocks again.  for each ancestor, get max scores of all
    #   ancestors and store feature as max ancestor score / max score
    features = np.zeros((len(blocks), 1))
    for k, block in enumerate(blocks):
        if len(block.features['ancestors']) == 0:
            features[k] = 0.0
        else:
            # check for valid ancestors and scores
            valid_scores = [scores[ancestor]
                for ancestor in block.features['ancestors']
                if valid_nodes[ancestor]]
            if len(valid_scores) == 0:
                features[k] = 0.0
            else:
                block_max = max(valid_scores)
                features[k] = max(block_max / max_score, 0.0)

    return features
readability_features.nfeatures = 1


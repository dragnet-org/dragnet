"""
The Levenshtein distance code was originally taken from (retrieved June 21, 2012):
   http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/

It may eventually be updated to use different scores for insertions, deletions,
transpositions, etc. For the time being, however, it remains as presented in
the article.
"""
from __future__ import division
import os
import pkgutil

import joblib
from sklearn.pipeline import FeatureUnion, make_union

from .compat import model_path, range_, string_, PY2
from .features import get_feature


def dameraulevenshtein(seq1, seq2):
    """Calculate the Damerau-Levenshtein distance between sequences.

    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second. Although generally used with strings, any sequences of
    comparable objects will work.

    Transpositions are exchanges of *consecutive* characters; all other
    operations are self-explanatory.

    This implementation is O(N*M) time and O(M) space, for N and M the
    lengths of the two sequences.

    >>> dameraulevenshtein('ba', 'abc')
    2
    >>> dameraulevenshtein('fee', 'deed')
    2

    It works with arbitrary sequences too:
    >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
    2
    """
    # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
    # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
    # However, only the current and two previous rows are needed at once,
    # so we only store those.
    oneago = None
    thisrow = list(range_(1, len(seq2) + 1)) + [0]
    for x in range_(len(seq1)):
        # Python lists wrap around for negative indices, so put the
        # leftmost column at the *end* of the list. This matches with
        # the zero-indexed strings and saves extra calculation.
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in range_(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1] and
                    seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


def evaluation_metrics(predicted, actual, bow=True):
    """
    Input:
        predicted, actual = lists of the predicted and actual tokens
        bow: if true use bag of words assumption
    Returns:
        precision, recall, F1, Levenshtein distance
    """
    if bow:
        p = set(predicted)
        a = set(actual)

        true_positive = 0
        for token in p:
            if token in a:
                true_positive += 1
    else:
        # shove actual into a hash, count up the unique occurances of each token
        # iterate through predicted, check which occur in actual
        from collections import defaultdict
        act = defaultdict(lambda: 0)
        for token in actual:
            act[token] += 1

        true_positive = 0
        for token in predicted:
            if act[token] > 0:
                true_positive += 1
                act[token] -= 1

        # for shared logic below
        p = predicted
        a = actual

    try:
        precision = true_positive / len(p)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_positive / len(a)
    except ZeroDivisionError:
        recall = 0.0
    try:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0.0

    # return (precision, recall, f1, dameraulevenshtein(predicted, actual))
    return (precision, recall, f1)


def get_and_union_features(features):
    """
    Get and combine features in a :class:`FeatureUnion`.

    Args:
        features (str or List[str], ``Features`` or List[``Features``], or List[Tuple[str, ``Features``]]):
            One or more features to be used to transform blocks into a matrix of
            numeric values. If more than one, a :class:`FeatureUnion` is
            automatically constructed. Example inputs::

                features = 'weninger'
                features = ['weninger', 'kohlschuetter']
                features = WeningerFeatures()
                features = [WeningerFeatures(), KohlschuetterFeatures()]
                features = [('weninger', WeningerFeatures()), ('kohlschuetter', KohlschuetterFeatures())]

    Returns:
        :class:`FeatureUnion` or ``Features``
    """
    if not features:
        raise ValueError('invalid `features`: may not be null')
    if isinstance(features, (list, tuple)):
        if isinstance(features[0], tuple):
            return FeatureUnion(features)
        elif isinstance(features[0], string_):
            return FeatureUnion([(feature, get_feature(feature)) for feature in features])
        else:
            return make_union(*features)
    elif isinstance(features, string_):
        return get_feature(features)
    else:
        return features

def load_pickled_model(filename, dirname=None):
    """
    Load a pickled ``Extractor`` model from disk.

    Args:
        filename (str): Name of pickled model file under ``dirname``.
        dirname (str): Name of directory on disk containing the pickled model.
            If None, dragnet's default pickled model directory is used:
            /path/to/dragnet/pickled_models/[PY_VERSION]_[SKLEARN_VERSION]

    Returns:
        :class:`dragnet.extractor.Extractor`
    """
    if dirname is None:
        pkg_filename = pkgutil.get_loader('dragnet').get_filename('dragnet')
        pkg_dirname = os.path.dirname(pkg_filename)
        dirname = os.path.join(pkg_dirname, 'pickled_models', model_path)
    filepath = os.path.join(dirname, filename)
    return joblib.load(filepath)

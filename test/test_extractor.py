import io
import os

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from dragnet import Extractor
from dragnet.blocks import TagCountNoCSSReadabilityBlockifier
from dragnet.util import get_and_union_features
from dragnet.compat import str_cast


@pytest.fixture(scope="module")
def html():
    fname = os.path.join("test", "datafiles", "models_testing.html")
    with io.open(fname, mode="rt") as f:
        html_ = f.read()
    return html_


def test_extractor(html):
    prob_threshold = 0.5
    blockifier = TagCountNoCSSReadabilityBlockifier()
    features = get_and_union_features(['weninger', 'kohlschuetter', 'readability'])
    # initialize model from pre-fit attributes
    model_attrs = {
        'C': 1.0,
        'class_weight': None,
        'classes_': [0, 1],
        'coef_': [[0.00501458328421719, -0.0006331822163374379, -0.6699789320373452, 0.026069227973339763, -1.5552477377277252, 0.02980432745983307, -0.965575689884716, 0.019509367890934326, -0.35692924115362307]],
        'dual': False,
        'fit_intercept': True,
        'intercept_': [-1.2071425754440765],
        'intercept_scaling': 1,
        'max_iter': 100,
        'multi_class': 'ovr',
        'n_iter_': [12],
        'n_jobs': 1,
        'penalty': 'l2',
        'solver': 'liblinear',
        'tol': 0.0001,
        'warm_start': False}
    model = LogisticRegression()
    for k, v in model_attrs.items():
        if isinstance(v, list):
            setattr(model, k, np.array(v))
        else:
            setattr(model, k, v)

    # extract content via the extractor class
    extractor = Extractor(blockifier, features=features, model=model,
                          to_extract='content', prob_threshold=prob_threshold)
    extractor_content = extractor.extract(html)

    # extract content via individual components
    blocks = blockifier.blockify(html)
    features_mat = features.transform(blocks)
    positive_idx = list(model.classes_).index(1)
    preds = (model.predict_proba(features_mat) > prob_threshold)[:, positive_idx].astype(int)
    components_content = '\n'.join(str_cast(blocks[ind].text) for ind in np.flatnonzero(preds))

    assert extractor_content is not None
    assert extractor_content == components_content

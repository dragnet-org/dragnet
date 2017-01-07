import gzip
import os
import pkgutil
import warnings

from sklearn import __version__ as sklearn_version

from .compat import PY2, pickle, bytes_io, sklearn_path
from .blocks import TagCountNoCSSBlockifier, TagCountNoCSSReadabilityBlockifier
from .content_extraction_model import baseline_model
from .content_extraction_model import ContentCommentsExtractionModel, SklearnWrapper
from .weninger import Weninger


def _load_pickled_model(fname, compressed='gzip'):
    model_bytes = pkgutil.get_data(
        'dragnet', os.path.join('pickled_models', sklearn_path, fname))
    if compressed == 'gzip':
        with gzip.GzipFile(fileobj=bytes_io(model_bytes), mode='rb') as f:
            return pickle.load(f)
            # if PY3: pickle.load(f, encoding='bytes') ?
    else:
        return pickle.loads(model_bytes)


try:
    # make instances
    content_extractor = _load_pickled_model(
        'kohlschuetter_weninger_readability_content_model.pickle.gz',
        compressed='gzip')
    content_comments_extractor = _load_pickled_model(
        'kohlschuetter_weninger_readability_content_comments_model.pickle.gz',
        compressed='gzip')

    # monkey patches
    content_extractor._blockifier = TagCountNoCSSReadabilityBlockifier
    content_extractor._block_model = SklearnWrapper(content_extractor._block_model)
    content_comments_extractor._blockifier = TagCountNoCSSReadabilityBlockifier
    content_comments_extractor._block_model = SklearnWrapper(
        content_comments_extractor._block_model)

    # finally make the model that returns both main content and content+comments
    content_and_content_comments_extractor = ContentCommentsExtractionModel(
        content_extractor._blockifier, content_extractor._features,
        content_extractor._block_model, content_comments_extractor._block_model,
        content_extractor._threshold)
except Exception as e:
    warnings.warn(
        "Unable to unpickle ContentExtractionModel! "
        "Your version of scikit-learn ({}) may not be compatible. "
        "Setting extractors to None.".format(sklearn_version),
        UserWarning)
    content_extractor = None
    content_comments_extractor = None
    content_and_content_comments_extractor = None

# weninger_model = Weninger()
# kohlschuetter_model = pickle.loads(
#     pkgutil.get_data('dragnet', 'pickled_models/kohlschuetter_1.0_content_model.pickle'))
# kohlschuetter_css_model = pickle.loads(
#     pkgutil.get_data('dragnet', 'pickled_models/kohlschuetter_css_10.0_content_model.pickle'))
# kohlschuetter_css_weninger_model = pickle.loads(
#     pkgutil.get_data('dragnet', 'pickled_models/kohlschuetter_css_weninger_100.0_content_model.pickle'))
# kohlschuetter_weninger_model = pickle.loads(
#     pkgutil.get_data('dragnet', 'pickled_models/kohlschuetter_weninger_1.0_content_model.pickle'))
#
# # monkey patch the blockifiers to eliminate CSS features when not needed
# weninger_model._blockifier = TagCountNoCSSBlockifier
# kohlschuetter_model._blockifier = TagCountNoCSSBlockifier
# kohlschuetter_weninger_model._blockifier = TagCountNoCSSBlockifier

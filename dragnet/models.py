import gzip
import os
import pkgutil
import warnings

from sklearn import __version__ as sklearn_version

from .compat import pickle, bytes_io, model_path
from .blocks import TagCountNoCSSReadabilityBlockifier
from .content_extraction_model import baseline_model
from .content_extraction_model import ContentCommentsExtractionModel, SklearnWrapper


def _load_pickled_model(fname, compressed='gzip'):
    model_bytes = pkgutil.get_data(
        'dragnet', os.path.join('pickled_models', model_path, fname))
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

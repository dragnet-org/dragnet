
import pkgutil
import cPickle as pickle
import os

from StringIO import StringIO
from gzip import GzipFile

from .blocks import TagCountNoCSSBlockifier, TagCountNoCSSReadabilityBlockifier
from .content_extraction_model import baseline_model
from .content_extraction_model import ContentCommentsExtractionModel, SklearnWrapper
from .weninger import Weninger

# make instances
with GzipFile(
    fileobj=StringIO(pkgutil.get_data(
        'dragnet',
         os.path.join('pickled_models',
            'kohlschuetter_weninger_readability_content_model.pickle.gz'))),
    mode='r') as fin:
    content_extractor = pickle.load(fin)

with GzipFile(
    fileobj=StringIO(pkgutil.get_data(
        'dragnet', 
         os.path.join('pickled_models', 
            'kohlschuetter_weninger_readability_content_comments_model.pickle.gz'))),
    mode='r') as fin:
    content_comments_extractor = pickle.load(fin)

weninger_model = Weninger()
kohlschuetter_model = pickle.loads(
    pkgutil.get_data('dragnet', 'pickled_models/kohlschuetter_1.0_content_model.pickle'))
kohlschuetter_css_model = pickle.loads(
    pkgutil.get_data('dragnet', 'pickled_models/kohlschuetter_css_10.0_content_model.pickle'))
kohlschuetter_css_weninger_model = pickle.loads(
    pkgutil.get_data('dragnet', 'pickled_models/kohlschuetter_css_weninger_100.0_content_model.pickle'))
kohlschuetter_weninger_model = pickle.loads(
        pkgutil.get_data('dragnet', 'pickled_models/kohlschuetter_weninger_1.0_content_model.pickle'))


# monkey patch the blockifiers to eliminate CSS features when not needed
weninger_model._blockifier = TagCountNoCSSBlockifier
kohlschuetter_model._blockifier = TagCountNoCSSBlockifier
kohlschuetter_weninger_model._blockifier = TagCountNoCSSBlockifier
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


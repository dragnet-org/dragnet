
import pickle
import pkgutil

from .blocks import TagCountNoCSSBlockifier
from .content_extraction_model import baseline_model
from .weninger import Weninger

# make instances
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


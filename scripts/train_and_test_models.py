import argparse
import os
import sys

from sklearn.ensemble import ExtraTreesClassifier

#from dragnet.model_training import evaluate_models_tokens, train_models
from dragnet.model_training import train_model
from dragnet.extractor import Extractor


MODEL = ExtraTreesClassifier(
    n_estimators=10, n_jobs=1, oob_score=False, bootstrap=False,
    class_weight=None, criterion='gini', max_depth=None, max_features=None,
    max_leaf_nodes=None, min_samples_leaf=75, min_samples_split=2,
    min_weight_fraction_leaf=0.0, random_state=None, verbose=0, warm_start=None)
"""
``sklearn`` model used to extract content (and comments) from featurized blocks
"""


def main():

    parser = argparse.ArgumentParser(
        description='Script to train and test a dragnet ContentExtractionModel')
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='root directory for all training data, e.g. /path/to/dragnet_data')
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='directory to which models, training errors, etc. will be saved')
    parser.add_argument(
        '--content_or_comments', type=str, required=True,
        choices=['content', 'both'],
        help="""type of information to be extracted by the model: just "content",
             or "both" content and comments""")
    parser.add_argument(
        '--features', type=str, nargs='+',
        choices=['kohlschuetter', 'weninger', 'readability'],
        default=['kohlschuetter', 'weninger', 'readability'],
        help="""the name(s) of one or more features to use as model inputs; must
             be one of the features known by `dragnet.AllFeatures`""")
    args = vars(parser.parse_args())

    # train the model
    #dragnet_model = train_models(
    #    args['data_dir'], args['output_dir'], args['features'], MODEL,
    #    content_or_comments=args['content_or_comments'])
    if args['content_or_comments'] == 'content':
        to_extract = 'content'
    elif args['content_or_comments'] == 'both':
        to_extract = ['content', 'comments']
    extractor = Extractor(features=args['features'], model=MODEL, to_extract=to_extract)
    trained_extractor = train_model(extractor, args['data_dir'], args['output_dir'])

    # and evaluate it
    #figname_prefix = '_'.join(args['features']) + \
    #    '_content_' if args['content_or_comments'] == 'content' else '_content_comments_'
    #evaluate_models_tokens(
    #    args['data_dir'], dragnet_model,
    #    content_or_comments=args['content_or_comments'],
    #    figname_root=os.path.join(args['output_dir'], figname_prefix))


if __name__ == '__main__':
    sys.exit(main())

from __future__ import division, print_function

import io
import logging
import os
import pprint
import numpy as np

from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion

from .blocks import simple_tokenizer
from .compat import GridSearchCV, model_path, string_, train_test_split, str_cast
from .data_processing import prepare_all_data
from .util import dameraulevenshtein


def evaluate_model_predictions(y_true, y_pred, weights=None):
    """
    Evaluate the performance of an extractor model's binary classification
    predictions, typically at the block level, of whether a block is content
    or not.

    Args:
        y_true (``np.ndarray``)
        y_pred (``np.ndarray``)
        weights (``np.ndarray``)

    Returns:
        Dict[str, float]
    """
    if isinstance(y_pred[0], np.ndarray):
        y_pred = np.concatenate(y_pred)
    if isinstance(y_true[0], np.ndarray):
        y_true = np.concatenate(y_true)
    if (weights is not None) and (isinstance(weights[0], np.ndarray)):
        weights = np.concatenate(weights)

    accuracy = accuracy_score(
        y_true, y_pred, normalize=True, sample_weight=weights)
    precision = precision_score(
        y_true, y_pred, average='binary', pos_label=1, sample_weight=weights)
    recall = recall_score(
        y_true, y_pred, average='binary', pos_label=1, sample_weight=weights)
    f1 = f1_score(
        y_true, y_pred, average='binary', pos_label=1, sample_weight=weights)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def evaluate_extracted_tokens(gold_content, extr_content):
    """
    Evaluate the similarity between gold-standard and extracted content,
    typically for a single HTML document, as another way of evaluating the
    performance of an extractor model.

    Args:
        gold_content (str or Sequence[str]): Gold-standard content, either as a
            string or as an already-tokenized list of tokens.
        extr_content (str or Sequence[str]): Extracted content, either as a
            string or as an already-tokenized list of tokens.

    Returns:
        Dict[str, float]
    """
    if isinstance(gold_content, string_):
        gold_content = simple_tokenizer(gold_content)
    if isinstance(extr_content, string_):
        extr_content = simple_tokenizer(extr_content)
    gold_set = set(gold_content)
    extr_set = set(extr_content)
    jaccard = len(gold_set & extr_set) / len(gold_set | extr_set)
    levenshtein = dameraulevenshtein(gold_content, extr_content)
    return {'jaccard': jaccard, 'levenshtein': levenshtein}


def train_model(extractor, data_dir, output_dir=None):
    """
    Train an extractor model, then write train/test block-level classification
    performance as well as the model itself to disk in ``output_dir``.

    Args:
        extractor (:class:`Extractor`): Instance of the ``Extractor`` class to
            be trained.
        data_dir (str): Directory on disk containing subdirectories for all
            training data, including raw html and gold standard blocks files
        output_dir (str): Directory on disk to which the trained model files,
            errors, etc. are to be written. If None, outputs are not saved.

    Returns:
        :class:`Extractor`: A trained extractor model.
    """
    # set up directories and file naming
    output_dir, fname_prefix = _set_up_output_dir_and_fname_prefix(output_dir, extractor)

    # prepare and split the data
    logging.info('preparing, splitting, and concatenating the data...')
    data = prepare_all_data(data_dir)
    training_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42)
    train_html, train_labels, train_weights = extractor.get_html_labels_weights(training_data)
    test_html, test_labels, test_weights = extractor.get_html_labels_weights(test_data)

    # fit the extractor on training data
    # then evaluate it on train and test data
    logging.info('fitting and evaluating the extractor features and model...')
    try:
        extractor.fit(train_html, train_labels, weights=train_weights)
    except (TypeError, ValueError):
        extractor.fit(train_html, train_labels)
    train_eval = evaluate_model_predictions(
        np.concatenate(train_labels), extractor.predict(train_html),
        np.concatenate(train_weights))
    test_eval = evaluate_model_predictions(
        np.concatenate(test_labels), extractor.predict(test_html),
        np.concatenate(test_weights))

    # report model performance
    _report_model_performance(output_dir, fname_prefix, train_eval, test_eval)

    # pickle the final model
    _write_model_to_disk(output_dir, fname_prefix, extractor)

    return extractor


def train_many_models(extractor, param_grid, data_dir, output_dir=None,
                      **kwargs):
    """
    Train many extractor models, then for the best-scoring model, write
    train/test block-level classification performance as well as the model itself
    to disk in ``output_dir``.

    Args:
        extractor (:class:`Extractor`): Instance of the ``Extractor`` class to
            be trained.
        param_grid (dict or List[dict]): Dictionary with parameters names (str)
            as keys and lists of parameter settings to try as values, or a list
            of such dictionaries, in which case the grids spanned by each are
            explored. See documentation for :class:`GridSearchCV` for details.
        data_dir (str): Directory on disk containing subdirectories for all
            training data, including raw html and gold standard blocks files
        output_dir (str): Directory on disk to which the trained model files,
            errors, etc. are to be written. If None, outputs are not saved.
        **kwargs:
            scoring (str or Callable): default 'f1'
            cv (int): default 5
            n_jobs (int): default 1
            verbose (int): default 1

    Returns:
        :class:`Extractor`: The trained extractor model with the best-scoring
            set of params.

    See Also:
        Documentation for grid search :class:`GridSearchCV` in ``scikit-learn``:
            http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    """
    # set up directories and file naming
    output_dir, fname_prefix = _set_up_output_dir_and_fname_prefix(output_dir, extractor)

    # prepare and split the data
    logging.info('preparing and splitting the data...')
    data = prepare_all_data(data_dir)
    training_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42)
    train_html, train_labels, train_weights = extractor.get_html_labels_weights(training_data)
    test_html, test_labels, test_weights = extractor.get_html_labels_weights(test_data)

    # filter docs we can't get features from
    train_blocks = np.array([extractor.blockifier.blockify(doc)
                            for doc in train_html])
    train_mask = [extractor._has_enough_blocks(blocks) for blocks in train_blocks]
    train_blocks = train_blocks[train_mask]
    train_labels = np.concatenate(train_labels[train_mask])
    train_weights = np.concatenate(train_weights[train_mask])
    test_labels = np.concatenate(test_labels)
    test_weights = np.concatenate(test_weights)
    # get features
    # TODO: This only 'fit's one doc at a time. No feature fitting actually
    # happens for now, but this might be important if the features change
    train_features = np.concatenate([extractor.features.fit_transform(blocks)
                                    for blocks in train_blocks])

    # fit many models
    gscv = GridSearchCV(
        extractor.model, param_grid, fit_params={'sample_weight': train_weights},
        scoring=kwargs.get('scoring', 'f1'), cv=kwargs.get('cv', 5),
        n_jobs=kwargs.get('n_jobs', 1), verbose=kwargs.get('verbose', 1))
    gscv = gscv.fit(train_features, train_labels)

    logging.info('Score of the best model, on left-out data: %s', gscv.best_score_)
    logging.info('Params of the best model: %s', gscv.best_params_)

    # evaluate best model on train and test data
    extractor.model = gscv.best_estimator_
    train_eval = evaluate_model_predictions(
        train_labels, extractor.predict(train_html[train_mask]), weights=train_weights)
    test_eval = evaluate_model_predictions(
        test_labels, extractor.predict(test_html), weights=test_weights)

    # pickle the final model
    _write_model_to_disk(output_dir, fname_prefix, extractor)

    return extractor


def _set_up_output_dir_and_fname_prefix(output_dir, extractor):
    if output_dir is not None:
        output_dir = os.path.join(output_dir, model_path)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if isinstance(extractor.features, FeatureUnion):
            fname_prefix = '_'.join(sorted(f[0] for f in extractor.features.transformer_list))
        elif hasattr(extractor.features, '__name__'):
            fname_prefix = extractor.features.__name__
        else:
            fname_prefix = str(extractor.features)
        fname_prefix += '_' + '_'.join(sorted(extractor.to_extract))
    else:
        fname_prefix = ''
    return output_dir, fname_prefix


def _report_model_performance(output_dir, fname_prefix, train_eval, test_eval):
    if output_dir is not None:
        output_fname = os.path.join(output_dir, fname_prefix + '_block_errors.txt')
        logging.info('writing evaluation metrics to file: %s', output_fname)
        with io.open(output_fname, mode='w') as f:
            f.write(u'Training errors for final model (block level):\n')
            f.write(str_cast(pprint.pformat(train_eval)))
            f.write(u'\nTest errors for final model (block level):\n')
            f.write(str_cast(pprint.pformat(test_eval)))
    # or just print it out
    else:
        print('Training errors for final model (block level):\n')
        pprint.pprint(train_eval)
        print('\nTest errors for final model (block level):\n')
        pprint.pprint(test_eval)


def _write_model_to_disk(output_dir, fname_prefix, extractor):
    if output_dir is not None:
        output_fname = os.path.join(output_dir, fname_prefix + '_model.pkl.gz')
        logging.info('writing model to file: %s', output_fname)
        joblib.dump(extractor, output_fname, compress=3)

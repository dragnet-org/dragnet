
import re
import json
import numpy as np
import pylab as plt
import glob
import codecs

from mozsci.cross_validate import cv_kfold

from .blocks import Blockifier
from . import evaluation_metrics
from .content_extraction_model import ContentExtractionModel
from .data_processing import simple_tokenizer, DragnetModelData, read_gold_standard, get_list_all_corrected_files, read_HTML_file, add_plot_title


def run_score_content_detection(html, gold_standard, content_detector, tokenizer=simple_tokenizer):
    """
    Input:
        html = html string
        gold_standard = tokens of gold standard
        content_detector = callable with interface content = content_detector(html)
        tokenizer
    Output:
        the scoring metrics
    """
    content = content_detector(html)
    content_tokens = tokenizer(content)
    return evaluation_metrics(content_tokens, gold_standard, bow=False)



class DragnetModelTrainer(object):
    def __init__(self, tokenizer=simple_tokenizer, content_or_comments='both', kfolds=5, weighted=False):
        """
        tokenizer = callable with interface
            list_of_tokens = tokenizer(string)
        content_or_comments = one of 'comments', 'content' or 'both'
        """
        self.tokenizer = tokenizer
        if content_or_comments not in ('content', 'comments', 'both'):
            raise InputError("invalid content_or_comments")
        self.content_or_comments = content_or_comments
        self.kfolds = kfolds
        self.weighted = weighted

    def _get_labels(self, content, comments):
        # content, comments = the list tuple elements in data.training_data
        if self.content_or_comments == 'content':
            return content
        elif self.content_or_comments == 'comments':
            return comments
        else:
            # content, comments are
            # (list of block 0/1 flag, list of # tokens, all tokens as a list)
            return (np.logical_or(content[0], comments[0]),
                    content[1],
                    content[2] + comments[2])

    def make_features_from_data(self, data, model, training_or_test='training', return_blocks=False, train=False):
        """Given the data and a model, make the features
        using model.make_features

        uses self.comments_or_content
        If return_blocks = True then also return the block level strings
        train = passed into model.make_features"""

        # get the features from the first document to see how many features we have
        if training_or_test == 'training':
            data_for_features = data.training_data
        elif training_or_test == 'test':
            data_for_features = data.test_data
        else:
            raise ValueError

        first_data_features, blocks = model.make_features(
                data_for_features[0][0], train, encoding=data_for_features[0][3])
        features = np.zeros((0, first_data_features.shape[1]))
        labels = np.zeros((0,))
        weights = np.zeros((0,))   # the token counts in the blocks

        all_blocks = []

        for html, content, comments, encoding in data_for_features:
            training_features, blocks = model.make_features(html, train, encoding=encoding)
            if training_features is None:
                # document is too short -- has < blocks.
                # skip it
                continue
            features = np.vstack((features, training_features))
            this_labels, this_weight, this_tokens = self._get_labels(content, comments)
            weights = np.hstack((weights, this_weight))
            labels = np.hstack((labels, this_labels))
            if return_blocks:
                all_blocks.extend(blocks)

        labels = labels.astype(np.int)

        if features.shape[0] != len(labels) or len(labels) != len(weights):
            raise ValueError("Number of features, labels and weights do not match!")

        if not return_blocks:
            return features, labels, weights
        else:
            return features, labels, weights, all_blocks
           


    def train_model(self, data, model_library, model):
        """data is an instance of DragnetModelData
        model_library is a list of model definitions as input to
         run_train_models
        model provides model.make_features to make the features
        """
        from mozsci.map_train import run_train_models

        # to train the model need a set of all the features and their labels
        # the features + labels are block level

        # get the features from the first document to see how many features we have
        features, labels, weights = self.make_features_from_data(data, model)

        # cap weights!
        weights = np.minimum(weights, 200)

        # do kfold cross validation
        folds = cv_kfold(len(labels), self.kfolds, seed=2)

        if self.weighted:
            errors = run_train_models(processes=4, model_library=model_library,
                X=features, y=labels, folds=folds, weights=weights)
        else:
            errors = run_train_models(processes=4, model_library=model_library,
                X=features, y=labels, folds=folds)

        return errors, features, labels, weights, folds



def plot_errors(errors, reg_parm_str):
    """reg_parm_str the key for the kwargs that defines the regularization
    coef
    
    e.g. 'C', 'lam', 'minsize' for liblinear, logistic regression, ClassTree"""

    # train/test X error number X error type
    # error type is accuracy, auc, f1, precision, recall in order
    errors_plot = np.zeros((2, len(errors), 5))  
    reg_parm = np.zeros(len(errors))
    k = 0
    for model, err in errors.iteritems():
        # get the regularization parameter for this model
        c = float(re.search("'" + reg_parm_str + "':\s*([\.0-9]+)[^}]*}\]", model).group(1))
        reg_parm[k] = c

        varn = 0
        for var in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
            errors_plot[0, k, varn] = err['train'][var]
            errors_plot[1, k, varn] = err['test'][var]
            varn += 1

        k += 1

    # plot
    idreg = reg_parm.argsort()
    reg_parm = reg_parm[idreg]
    errors_plot = errors_plot[:, idreg, :]

    vars = ['accuracy', 'auc', 'f1', 'precision', 'recall']
    label = ['train', 'test']
    fig = plt.figure(1)
    fig.clf()
    for k in xrange(2):
        varn = 0
        for varn in xrange(len(vars)):
            plt.subplot(230 + varn + 1)
            plt.plot(np.log(reg_parm), errors_plot[k, :, varn], label=label[k])
            plt.title(vars[varn])
            varn += 1
        if k == 1 and varn == 5:
            plt.legend(loc='lower right')
    fig.show()


def accuracy_auc(y, ypred, weights=None):
    """Compute the accuracy, AUC, precision, recall and f1"""
    from mozsci.evaluation import classification_error, auc_wmw_fast, precision_recall_f1
    prf1 = precision_recall_f1(y, ypred, weights=weights)
    return { 'accuracy':1.0 - classification_error(y, ypred, weights=weights),
             'auc':auc_wmw_fast(y, ypred, weights=weights),
             'precision':prf1[0],
             'recall':prf1[1],
             'f1':prf1[2] }


def evaluate_models_tokens(datadir, dragnet_model, figname_root=None,
    tokenizer=simple_tokenizer, cetr=False, content_or_comments='both'):
    """
    Evaluate a trained model on the token level.

    datadir = all the data lives here
    dragnet_model = Implements the ContentExtractionModel interface,
        specifically the analyze method
    if figname_root is not None, then output plots/tables to
        figname_root + extension
    tokenizer = implements tokenizer interface
    cetr = if True, then handle the gold standard in CETR format (parse it)
    content_or_comments = 'content', 'comments' or 'both' to specify
        what to include in the gold standard

    Outputs:
        saves png files
    returns scores
    """
    all_files = get_list_all_corrected_files(datadir)

    gold_standard_tokens = {}
    for fname, froot in all_files:
        content, comments = read_gold_standard(datadir, froot, cetr)
        if content_or_comments == 'content':
            gold = content
        elif content_or_comments == 'comments':
            gold = comments
        elif content_or_comments == 'both':
            gold = content + ' ' + comments
        else:
            raise ValueError("Invalid input for content_or_comments")
        tokens = tokenizer(gold)
        tokens = [token.encode('utf-8') for token in tokens]
        if len(tokens) > 0:
            gold_standard_tokens[froot] = tokens

    use_list = type(dragnet_model) == list
    if use_list:
        errors = np.zeros((len(gold_standard_tokens), 3, len(dragnet_model)))
    else:
        errors = np.zeros((len(gold_standard_tokens), 3))

    k = 0
    for froot, gstok in gold_standard_tokens.iteritems():
        html, encoding = read_HTML_file(datadir, froot)
        if use_list:
            for i in xrange(len(dragnet_model)):
                # make an analyze function to handle the encoding
                dm = lambda x: dragnet_model[i].analyze(x, encoding=encoding)
                errors[k, :, i] = run_score_content_detection(
                    html, gstok, dm, tokenizer=tokenizer)
        else:
            dm = lambda x: dragnet_model.analyze(x, encoding=encoding)
            errors[k, :] = run_score_content_detection(
                    html, gstok, dragnet_model.analyze, tokenizer=tokenizer)
        k += 1


    # make some plots
    # if just a single model, plot histograms of precision, recall, f1
    ti = ['Precision', 'Recall', 'F1', 'edit distance']
    if not use_list:
        fig = plt.figure(1)
        fig.clf()

        for k in xrange(3):
            plt.subplot(2, 2, k + 1)
            plt.hist(errors[:, k], 20)
            plt.title("%s %s" % (ti[k], np.mean(errors[:, k])))

        add_plot_title("Token level evaluation")
        fig.show()

        if figname_root is not None:
            fig.savefig(figname_root + '.png')

    else:
        # multiple models
        thresholds = [ele._threshold for ele in dragnet_model]
        i = 1

        if figname_root is not None:
            ftable = open(figname_root + '_scores_errors.txt', 'w')

        fig = plt.figure(1)
        fig.clf()

        scores = errors

        avg_scores = scores.mean(axis=0)
        std_scores = scores.std(axis=0)

        for k in xrange(3):
            ax = plt.subplot(2, 2, k + 1)
            plt.plot(thresholds, avg_scores[k, :], 'b')
            plt.plot(thresholds, avg_scores[k, :] + std_scores[k, :], 'k--')
            plt.plot(thresholds, avg_scores[k, :] - std_scores[k, :], 'k--')
            plt.title(ti[k])
            ax.grid(True)

        fig.show()

        if figname_root is not None:
            fig.savefig(figname_root + '.png')

        # write a table
        if figname_root is not None:
            ftable.write("Threshold | Precision | Recall | F1\n")
            for k in xrange(len(thresholds)):
                ftable.write("%s            %5.3f    %5.3f   %5.3f\n" % (thresholds[k], avg_scores[0, k], avg_scores[1, k], avg_scores[2, k]))

            i += 1

        if figname_root is not None:
            ftable.close()

    return errors


def train_models(datadir, output_prefix, features_to_use, lam=10,
    content_or_comments='both'):
    """Train a content extraction model.

    Does feature centering, trains the logistic regression model,
    pickles the final model and writes the train/test block level errors
    to a file

    datadir = root directory for all the data
    output_prefix = write the trained model files, errors, etc
        to files starting with this.
    features_to_use = a list of the features to use.  Must be one of the features
        known by AllFeatures
    lambda = lambda regularization parameter for LogisticRegression
    """
    import pprint
    import pickle
    from . import AllFeatures
    from .blocks import TagCountBlockifier as Blkr

    from mozsci.models import LogisticRegression
    from mozsci.numpy_util import NumpyEncoder

    # assemble the features
    feature_instances = []
    for f in features_to_use:
        feature_instances.append(AllFeatures.get(f))

    # compute the mean/std and save them
    data = DragnetModelData(datadir)
    trainer = DragnetModelTrainer(content_or_comments=content_or_comments)

    print "Initializing features"
    k = 0
    for f in feature_instances:
        # check to see if this feature needs to be init
        # if so, then init it, take the return object and serialize to json
        if hasattr(f, 'init_params'):
            # initialize it
            model_init = ContentExtractionModel(Blkr, [f], None)
            features, labels, weights = trainer.make_features_from_data(data, model_init, train=True)
            mean_std = f.init_params(features)
            f.set_params(mean_std)
            with open("%s_mean_std_%s.json" % (output_prefix, features_to_use[k]), 'w') as fout:
                fout.write("%s" % json.dumps(mean_std, cls=NumpyEncoder))
        k += 1

    model_to_train = ContentExtractionModel(Blkr, feature_instances, None)

    # train the model
    print "Training the model"
    model = LogisticRegression(lam=lam)
    features, labels, weights = trainer.make_features_from_data(data,
                         model_to_train, training_or_test='training')
    model.fit(features, labels, weights=np.minimum(weights, 200))

    print "Checking errors"
    train_errors = accuracy_auc(labels, model.predict(features), weights=weights)

    # check errors on test set
    test_features, test_labels, test_weights = trainer.make_features_from_data(data,
                     model_to_train, training_or_test='test')
    test_weights = np.minimum(test_weights, 200.0)
    test_errors = accuracy_auc(test_labels, model.predict(test_features), weights=test_weights)

    # write errors to a file
    with open(output_prefix + '_block_errors.txt', 'w') as f:
        f.write("Training errors for final model (block level):\n")
        pprint.pprint(train_errors, f)
        f.write("Test errors (block level):\n")
        pprint.pprint(test_errors, f)

    # pickle the final model!
    # use the one with threshold = 0.5
    pickle.dump(ContentExtractionModel(Blkr,
                        feature_instances,
                        model,
                        threshold=0.5), open(output_prefix + '_content_model.pickle', 'w'))

    print "done!"




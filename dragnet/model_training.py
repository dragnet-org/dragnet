
import re
import json
import numpy as np
import pylab as plt
import glob
import codecs

from mozsci.evaluation import classification_error, auc_wmw_fast, precision_recall_f1
from mozsci.cross_validate import cv_kfold

from .blocks import Blockifier
from . import evaluation_metrics
from .features import all_features
from .content_extraction_model import ContentExtractionModel
from .data_processing import simple_tokenizer, DragnetModelData


def add_plot_title(ti_str):
    """Add a string as a title on top of a subplot"""
    plt.figtext(0.5, 0.94, ti_str, ha='center', color='black', weight='bold', size='large')


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
    return evaluation_metrics(content_tokens, gold_standard)



def _compute_score(data_to_score, html, model, tokenizer, model_list, k):
    """Private helper for Pool, needed to compute the the score
    for a training instance in parallel"""
    print k
    if model_list:
        ret = np.zeros((3, len(model)))
        for i in xrange(len(model)):
            ret[:, i] = run_score_content_detection(html, data_to_score[2], model[i].analyze, tokenizer)
    else:
        ret = run_score_content_detection(html, data_to_score[2], model.analyze, tokenizer)
    return k, ret


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

    def evaluate_model(self, data, model, use_pool=True):
        """data = a list of documents + evaluation as stored in DragnetModelData
            as in data.training_data or data.test_data
        model = uses the .analyze function
            can be either something with .analyze implemented,
            or a list of objects where each element if the list implements
            .analyze

        Returns:  if model is not a list, scores an nobs X 3 array with
            scores[k, :] = [precision, recall, f1] for data point k
            If model is a list returns a nobs X 3 X nmodels array with
            scores[k, :, i] the scores for data point k with model i
        """
        # make a list of all the errors
        # scores[k, :] = [precision, recall, f1, edit distance]
        if isinstance(model, list):
            model_list = True
            scores = np.zeros((len(data), 3, len(model)))
        else:
            model_list = False
            scores = np.zeros((len(data), 3))

        def store_scores(k, r):
            if model_list:  
                scores[k, :, :] = r
            else:
                scores[k, :] = r

        if use_pool:
            from multiprocessing import Pool
            p = Pool(processes=16)
    
            # use the pool to compute the scores
            k = 0
            results = []
    
            for html, content, comments in data:
                data_to_score = self._get_labels(content, comments)
                results.append(p.apply_async(_compute_score, (data_to_score, html, model, self.tokenizer, model_list, k)))
                k += 1
    
            p.close()
            p.join()
    
            # collect the results
            for result in results:
                k, r = result.get()
                store_scores(k, r)

        else:
            k = 0
            results = []
            for html, content, comments in data:
                data_to_score = self._get_labels(content, comments)
                tk, r = _compute_score(data_to_score, html, model, self.tokenizer, model_list, k)
                store_scores(k, r)
                k += 1

        return scores


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
            raise InputError

        first_data_features, blocks = model.make_features(data_for_features[0][0], train)
        features = np.zeros((0, first_data_features.shape[1]))
        labels = np.zeros((0,))
        weights = np.zeros((0,))   # the token counts in the blocks

        all_blocks = []

        for html, content, comments in data_for_features:
            training_features, blocks = model.make_features(html, train)
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
    """Compute both the accuracy and AUC"""
    prf1 = precision_recall_f1(y, ypred, weights=weights)
    return { 'accuracy':1.0 - classification_error(y, ypred, weights=weights),
             'auc':auc_wmw_fast(y, ypred, weights=weights),
             'precision':prf1[0],
             'recall':prf1[1],
             'f1':prf1[2] }


def evaluate_models_tokens(datadir, dragnet_model, figname_root=None,
    tokenizer=simple_tokenizer):
    """
    Evaluate a trained model on the token level.

    datadir = all the data lives here
    dragnet_model = passed into .evaluate_model in DragnetModelTrainer
    if figname_root is not None, then output plots to figname_root + training/test.png

    Outputs:
        saves png files
    returns (training_errors, test_errors)
    """

    # do an overall precision/recall test on TOKENS
    data = DragnetModelData(datadir)

    # test the data!
    trainer = DragnetModelTrainer(tokenizer)
    training_errors = trainer.evaluate_model(data.training_data, dragnet_model, use_pool=False)
    test_errors = trainer.evaluate_model(data.test_data, dragnet_model, use_pool=False)

    # make some plots
    # if just a single model, plot histograms of precision, recall, f1
    ti_scores = ['Training', 'Test']
    ti = ['Precision', 'Recall', 'F1', 'edit distance']
    if len(training_errors.shape) == 2:
        i = 1
        for scores in [training_errors, test_errors]:
            fig = plt.figure(i)
            fig.clf()

            for k in xrange(3):
                plt.subplot(2, 2, k + 1)
                plt.hist(scores[:, k], 20)
                plt.title("%s %s" % (ti[k], np.mean(scores[:, k])))

            add_plot_title(ti_scores[i - 1])
            fig.show()

            if figname_root is not None:
                fig.savefig(figname_root + '_' + ti_scores[i - 1] + '.png')

            i += 1

    else:
        # multiple models
        thresholds = [ele._threshold for ele in dragnet_model]
        i = 1

        if figname_root is not None:
            ftable = open(figname_root + '_scores_errors.txt', 'w')

        for scores in [training_errors, test_errors]:
            fig = plt.figure(i)
            fig.clf()

            avg_scores = scores.mean(axis=0)
            std_scores = scores.std(axis=0)

            for k in xrange(3):
                ax = plt.subplot(2, 2, k + 1)
                plt.plot(thresholds, avg_scores[k, :], 'b')
                plt.plot(thresholds, avg_scores[k, :] + std_scores[k, :], 'k--')
                plt.plot(thresholds, avg_scores[k, :] - std_scores[k, :], 'k--')
                plt.title(ti[k])
                ax.grid(True)

            add_plot_title(ti_scores[i - 1])
            fig.show()

            if figname_root is not None:
                fig.savefig(figname_root + '_' + ti_scores[i - 1] + '.png')

            # write a table
            if figname_root is not None:
                ftable.write(ti_scores[i - 1] + '\n')
                ftable.write("Threshold | Precision | Recall | F1\n")
                for k in xrange(len(thresholds)):
                    ftable.write("%s            %5.3f    %5.3f   %5.3f\n" % (thresholds[k], avg_scores[0, k], avg_scores[1, k], avg_scores[2, k]))
 
            i += 1

        if figname_root is not None:
            ftable.close()

    return training_errors, test_errors



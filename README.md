
Dragnet
=====================================

[![Build Status](https://api.travis-ci.org/seomoz/dragnet.png)](https://api.travis-ci.org/seomoz/dragnet.png)

Dragnet isn't interested in the shiny chrome or boilerplate dressing of a 
web page. It's interested in... 'just the facts.'  The machine learning
models in Dragnet extract the main article content and optionally
user generated comments from a web page.  They provide state
of the art performance on variety of test benchmarks.

For more information on our approach check out:

* Our paper [<i>Content Extraction Using Diverse Feature Sets</i>](dragnet_www2013.pdf?raw=true), published
at WWW in 2013, gives an overview of the machine learning approach.
* [This blog post](https://moz.com/devblog/dragnet-content-extraction-from-diverse-feature-sets/) explains the intuition behind the algorithms.

This project was originally inspired by 
Kohlschütter et al, [Boilerplate Detection using Shallow Text Features](http://www.l3s.de/~kohlschuetter/publications/wsdm187-kohlschuetter.pdf) and 
Weninger et al [CETR -- Content Extraction with Tag Ratios](http://web.engr.illinois.edu/~weninge1/cetr/), and more recently by [Readability](https://github.com/buriy/python-readability).

# GETTING STARTED

Depending on your use case, we provide two separate models to extract
just the main article content or the content and any user generated
comments.  Each model implements the `analyze` method that
takes an HTML string and returns the content string.

```python
import requests
from dragnet import content_extractor, content_comments_extractor

# fetch HTML
url = 'https://moz.com/devblog/dragnet-content-extraction-from-diverse-feature-sets/'
r = requests.get(url)

# get main article without comments
content = content_extractor.analyze(r.content)

# get article and comments
content_comments = content_comments_extractor.analyze(r.content)
```

We also provide some additional models in `dragnet.models` but
don't recommend their use for anything other than academic curiousity.

## A note about encoding

If you know the encoding of the document (e.g. from HTTP headers),
you can pass it down to the parser:

    content = content_extractor.analyze(html_string, encoding='utf-8')

Otherwise, we try to guess the encoding from a `meta` tag or specified
`<?xml encoding=".."?>` tag.  If that fails, we assume "UTF-8".

## Installing

```
pip install dragnet
```

Dragnet is written in Python (developed with 2.7, not tested on 3)
and built on the numpy/scipy/Cython numerical computing environment.
In addition we use <a href="http://lxml.de/">lxml</a> (libxml2)
for HTML parsing.

# Contributing

We love contributions!  We are especially looking for someone who would
like to work on a Python 3 port.  Open an issue, or fork/create a pull
request.

# More details about the code structure

Each of the models in `dragnet.models` implements the
content extraction model interface defined in `ContentExtractionModel`.
A content extraction model encapsulates a blockifier, some feature
extractors and a machine learning model.

A blockifier implements `blockify` that takes a HTML string and returns a list
of block objects.  A feature extractor is a callable that takes a list
of blocks and returns a numpy array of features `(len(blocks), nfeatures)`.
There is some additional optional functionality
to "train" the feature (e.g. estimate parameters needed for centering)
specified in `features.py`.  The machine learning model implements
the [scikits-learn](http://scikit-learn.org/stable/) interface (`predict` and `fit`) and is used to compute
the content/no-content prediction for each block.

# Training/test data

The training and test data is available at [dragnet_data](https://github.com/seomoz/dragnet_data).

# Training content extraction models

0.  Download the training data (see above).  In what follows `ROOTDIR` contains
the root of the `dragnet_data` repo, another directory with similar
structure (`HTML` and `Corrected` sub-directories).
1.  Create the block corrected files needed to do supervised learning on the block level.
First make a sub-directory `$ROOTDIR/block_corrected/` for the output files, then run:

        from dragnet.data_processing import extract_gold_standard_all_training_data
        rootdir = '/path/to/dragnet_data/'
        extract_gold_standard_all_training_data(rootdir)

    This solves the longest common sub-sequence problem to determine
    which blocks were extracted in the gold standard.
    Occasionally this will fail if lxml (libxml2) cannot parse
    a HTML document.  In this case, remove the offending document and restart the process.
2.  Use k-fold cross validation in the training set to do model selection
    and set any hyperparameters.  Make decisions about the following:

    * Number of folds (recommend 5)
    * Whether to use just article content or content and comments.
    * The features to use
    * The machine learning model to use

    For example, to train the randomized decision tree classifier from
    sklearn using the shallow text features from Kohlschuetter et al.
    and the CETR features from Weninger et al.:

        from dragnet.model_training import DragnetModelTrainer, accuracy_auc
        from dragnet.data_processing import DragnetModelData
        from sklearn.ensemble import ExtraTreesClassifier

        datadir = '/path/to/dragnet_data/'

        kfolds = 5
        # recommend using weights but the model.fit methods needs to support it
        weighted = True
        features_to_use = ['kohlschuetter', 'weninger']

        content_or_comments = 'both'   # or 'content'

        model_library = [
            [ExtraTreesClassifier, accuracy_auc, None, (),
                {'n_estimators':10, 'max_features': None,
                 'min_samples_leaf':75}]
        ]

        data = DragnetModelData(datadir)
        trainer = DragnetModelTrainer(content_or_comments=content_or_comments,
            weighted=weighted, kfolds=kfolds)

        errors, features, labels, weights, folds = trainer.train_model(
            data, model_library, features_to_use)

    This trains the model and writes a pickled version of it along with some
    some *block level* classification errors to a file.
3.  Once you have decided on a final model, train it on the entire training
data using `dragnet.model_training.train_models`.
4.  As a last step, test the performance of the model on the test set (see
below).

## Evaluating content extraction models

Use `evaluate_models_tokens` in `model_training` to compute the token level
precision, recall and F1.  For example,
to evaluate the baseline model (keep everything) run:

    from dragnet.model_training import evaluate_models_tokens
    from dragnet.models import baseline_model

    rootdir = '/path/to/dragnet_data/'
    scores = evaluate_models_tokens(rootdir, baseline_model)


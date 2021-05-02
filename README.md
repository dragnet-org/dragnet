Dragnet
=======

[![Build Status](https://travis-ci.com/dragnet-org/dragnet.svg?branch=master)](https://travis-ci.com/dragnet-org/dragnet)

Dragnet isn't interested in the shiny chrome or boilerplate dressing
of a web page. It's interested in... 'just the facts.'  The machine
learning models in Dragnet extract the main article content and
optionally user generated comments from a web page.  They provide
state of the art performance on a variety of test benchmarks.

For more information on our approach check out:

* Our paper [_Content Extraction Using Diverse Feature Sets_](dragnet_www2013.pdf?raw=true), published
at WWW in 2013, gives an overview of the machine learning approach.
* [A comparison](https://moz.com/devblog/benchmarking-python-content-extraction-algorithms-dragnet-readability-goose-and-eatiht/) of Dragnet and alternate content extraction packages.
* [This blog post](https://moz.com/devblog/dragnet-content-extraction-from-diverse-feature-sets/) explains the intuition behind the algorithms.

This project was originally inspired by
Kohlschütter et al, [Boilerplate Detection using Shallow Text Features](http://www.l3s.de/~kohlschuetter/publications/wsdm187-kohlschuetter.pdf) and
Weninger et al [CETR -- Content Extraction with Tag Ratios](https://www3.nd.edu/~tweninge/cetr/#main-content-area), and more recently by [Readability](https://github.com/buriy/python-readability).

# GETTING STARTED

Depending on your use case, we provide two separate functions to extract
just the main article content or the content and any user generated
comments.  Each function takes an HTML string and returns the content string.

```python
import requests
from dragnet import extract_content, extract_content_and_comments

# fetch HTML
url = 'https://moz.com/devblog/dragnet-content-extraction-from-diverse-feature-sets/'
r = requests.get(url)

# get main article without comments
content = extract_content(r.content)

# get article and comments
content_comments = extract_content_and_comments(r.content)
```

We also provide a sklearn-style extractor class(complete with `fit` and 
`predict` methods). You can either train an extractor yourself, or load a
pre-trained one:
```python
from dragnet.util import load_pickled_model

content_extractor = load_pickled_model(
            'kohlschuetter_readability_weninger_content_model.pkl.gz')
content_comments_extractor = load_pickled_model(
            'kohlschuetter_readability_weninger_comments_content_model.pkl.gz')
            
content = content_extractor.extract(r.content)
content_comments = content_comments_extractor.extract(r.content)
```

## A note about encoding

If you know the encoding of the document (e.g. from HTTP headers),
you can pass it down to the parser:

```python
content = content_extractor.extract(html_string, encoding='utf-8')
```

Otherwise, we try to guess the encoding from a `meta` tag or specified
`<?xml encoding=".."?>` tag.  If that fails, we assume "UTF-8".

## Installing

Dragnet is written in Python (developed with 2.7, with support recently 
added for 3) and built on the numpy/scipy/Cython numerical computing
environment.
In addition we use [lxml](http://lxml.de/) (libxml2)
for HTML parsing.

We recommend installing from the master branch to ensure you have the latest
version.

### Installing with Docker:

This is the easiest method to install Dragnet and builds a Docker
container with Dragnet and its dependencies.

1. Install [Docker](https://docs.docker.com/get-docker/).
2. Clone the master branch: `git clone https://github.com/dragnet-org/dragnet.git`
3. Build the docker container: `docker build -t dragnet .`
4. Run the tests: `docker run dragnet make test`

You can also run an interactive Python session:
```bash
docker run -ti dragnet python3
```

### Installing without Docker

1.  Install the dependencies needed for Dragnet. The build depends on
GCC, numpy, Cython and lxml (which in turn depends on `libxml2`). We
use `provision.sh` to setup the dependencies in the Docker container,
so you can use it as a template and modify as appropriate for your
operation system.
2.  Clone the master branch: `git clone https://github.com/dragnet-org/dragnet.git`
3.  Install the requirements: `cd dragnet; pip install -r requirements.txt`
4.  Build dragnet:

```bash
$ cd dragnet
$ make install
# these should now pass
$ make test
```

# Contributing

We love contributions! Open an issue, or fork/create a pull
request.

# More details about the code structure

The `Extractor` class encapsulates a blockifier, some feature extractors and a machine learning model.

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

    ```python
    from dragnet.data_processing import extract_all_gold_standard_data
    rootdir = '/path/to/dragnet_data/'
    extract_all_gold_standard_data(rootdir)
    ```

    This solves the longest common sub-sequence problem to determine
    which blocks were extracted in the gold standard.
    Occasionally this will fail if lxml (libxml2) cannot parse
    a HTML document.  In this case, remove the offending document and restart
    the process.
2.  Use k-fold cross validation in the training set to do model selection
    and set any hyperparameters.  Make decisions about the following:

    * Whether to use just article content or content and comments.
    * The features to use
    * The machine learning model to use

    For example, to train the randomized decision tree classifier from
    sklearn using the shallow text features from Kohlschuetter et al.
    and the CETR features from Weninger et al.:

    ```python
    from dragnet.extractor import Extractor
    from dragnet.model_training import train_model
    from sklearn.ensemble import ExtraTreesClassifier

    rootdir = '/path/to/dragnet_data/'

    features = ['kohlschuetter', 'weninger', 'readability']

    to_extract = ['content', 'comments']   # or ['content']

    model = ExtraTreesClassifier(
        n_estimators=10,
        max_features=None,
        min_samples_leaf=75
    )
    base_extractor = Extractor(
        features=features,
        to_extract=to_extract,
        model=model
    )

    extractor = train_model(base_extractor, rootdir)
    ```

    This trains the model and, if a value is passed to `output_dir`, writes a
    pickled version of it along with some some *block level* classification
    errors to a file in the specified `output_dir`. If no `output_dir` is
    specified, the block-level performance is printed to stdout.
3.  Once you have decided on a final model, train it on the entire training
    data using `dragnet.model_training.train_models`.
4.  As a last step, test the performance of the model on the test set (see
    below).

## Evaluating content extraction models

Use `evaluate_models_predictions` in `model_training` to compute the token level
accuracy, precision, recall, and F1.  For example, to evaluate a trained model
run:

```python
from dragnet.compat import train_test_split
from dragnet.data_processing import prepare_all_data
from dragnet.model_training import evaluate_model_predictions

rootdir = '/path/to/dragnet_data/'
data = prepare_all_data(rootdir)
training_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

test_html, test_labels, test_weights = extractor.get_html_labels_weights(test_data)
train_html, train_labels, train_weights = extractor.get_html_labels_weights(training_data)

extractor.fit(train_html, train_labels, weights=train_weights)
predictions = extractor.predict(test_html)
scores = evaluate_model_predictions(test_labels, predictions, test_weights)
```

Note that this is the same evaluation that is run/printed in `train_model`

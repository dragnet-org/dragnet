
Dragnet
=====================================

# Overview

Dragnet isn't interested in the shiny chrome or boilerplate dressing of a 
webpage. It's interested in... 'just the facts.'

It is meant to become a collection of reference implementations of various
dechroming / content extraction algorithms.

This document contains details of the code and training data.
We also wrote a short paper describing the machine learning approach in Dragnet,
to be published at WWW 2013: TODO ADD LINK


# GETTING STARTED

## How to run a model

## Dependencies

Dragnet is written in Python (developed with 2.7, not tested on 3)
and built on the numpy/scipy/Cython numerical computing environment.
In addition we use matplotlib for visualizing the data and
<a href="http://lxml.de/">lxml</a> (libxml2)
for HTML parsing.  Finally, we use some of the utilities and models in
<a href="http://github.com/seomoz/mozsci">mozsci</a>.  You can install
the necessary packages with something like:

    apt-get install cython libxslt-dev libxml2-dev \
        python-numpy python-scipy python-matplotlib 
    pip install lxml

    # install mozsci
    git clone "git@github.com:seomoz/mozsci.git" 
    cd mozsci
    python setup.py install
    cd ..


# More details about the code structure

We provide a few high level classes for manipulating the data and doing the training.

`DragnetModelData` encapsulates the data set and includes methods for reading it from disk,
making some diagnostic plots and exposing the data to the
`DragnetModelTrainer` class.

The `DragnetModelTrainer` does ...


abstract base classes for blockifier, features and machine learning model
and a model class that chains them together and encapsuates all three


tokenizer = callable(string) and returns list of tokens


  blockifier = implements blockify that takes string and returns
           an list of Block instances

  features = callable that takes list of blocks
             and returns a numpy array of features (len(blocks), nfeatures)
             It accepts an optional keyword "train" that is only called in an initial
             pre-processing state for training
           feature.nfeatures = attribute that gives number of features
           optionally feature.init_params(features) that sets some global state
              if features.init_params is implemented, then must also implement
               features.set_params(ret) where ret is the returned value from
               features.init_params

  machine learning model = implements sklearn interface




# Details about the training data

A training data set consists of a collection of web pages and the extracted
'gold standard' content.  For our purposes we standardize  
a data set as a set of files on disk with a specific directory and naming
convention.  Each training example in the data set
is identified by a common file root.  
All the data for a given training example `X` lives under a common `ROOTDIR`
in a set of sub-directories as follows:

* `$ROOTDIR/HTML/` contains the raw HTML named `X.html`
* `$ROOTDIR/Corrected/` contains the extracted content named `X.html.corrected.txt`

We provide our training data described in the paper at TODO: add link

We have also tested our model on the data used in Weninger et al at TODO: add link.
and include a bash script `cetr_to_dragnet.sh` to convert the data
from CETR to Dragnet format.

## Creating your own training data

You can easily create your own training data:

1.  Create a directory hierarchy as detailed above (`HTML` and `Corrected` sub-directories)
2.  Add `HTML` and `Corrected` files.
    1.  Save HTML files to the directory to be used as training examples.  This is the raw HTML from crawling the page or "Save as.." in a browser.
    2.  Extract the content from the `html` files into the `Corrected` files.
        1.  Open each HTML file in a web browser with the network connection turned off
            and Javascript disabled.  This simulates the information available to a simple
            web crawler that does not execute Javascript or fetch additional
            resources other then the HTML.
        2.  Cut and paste any content into the `Corrected` text
            file.  If there are any comments, then separate the comments from the main
            article content in the text file with the string `!@#$%^&*()  COMMENTS`
            on its own line.
3.  Give your data back to the research community so everyone can benefit :-)

## Training content extraction models

1.  Create the block corrected files needed to do supervised learning on the block level.
First make a sub-directory `$ROOTDIR/block_corrected/` for the output files, then run:

        from dragnet.data_processing import extract_gold_standard_all_training_data
        rootdir = '/my/data/directory/'
        extract_gold_standard_all_training_data(rootdir)

    This solves the longest common sub-sequence problem to partition the data
    into blocks.  Occasionally this will fail if lxml (libxml2) cannot parse
    the HTML document.
2.  Run `split_data` to generate the `training.txt` and `test.txt` files split.


## Evaluating content extraction models

Use `evaluate_models_tokens` in `model_training` to evaluate models.  For example,
to evaluate the baseline model (keep everything) run:

    from dragnet.model_training import evaluate_models_tokens
    from dragnet.content_extraction_model import baseline_model

    rootdir = '/my/data/directory/'
    scores = evaluate_models_tokens(rootdir, baseline_model)



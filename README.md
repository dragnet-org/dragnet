
Dragnet
=====================================

Dragnet isn't interested in the shiny chrome or boilerplate dressing of a 
web page. It's interested in... 'just the facts.'

Our implementation is an ensemble of a few various
de-chroming / content extraction algorithms.

This document contains details of the code and training data.
We also wrote a short paper describing the machine learning approach in Dragnet,
to be published at WWW 2013.  You can find the paper
[here.](http://github.com/seomoz/dragnet/blob/master/dragnet_www2013.pdf?raw=true)

This project was heavily inspired by
Kohlschütter et al, [Boilerplate Detection using Shallow Text Features](http://www.l3s.de/~kohlschuetter/publications/wsdm187-kohlschuetter.pdf) and 
Weninger et al [CETR -- Content Extraction with Tag Ratios](http://web.engr.illinois.edu/~weninge1/cetr/).

# GETTING STARTED

We provide a set of models in `dragnet.models`.  Each implements the
`analyze` method that takes an HTML string and returns the content string.
For example, to run our implementation of Kohlschütter et al.
trained on our data,

    from dragnet.models import kohlschuetter_model
    content = kohlschuetter_model.analyze(html_string)

In addition we provide:

* `weninger_model`: the CETR k-means model from Weninger et al
* `kohlschuetter_css_model`: the shallow text + CSS features model from the paper
* `kohlschuetter_css_weninger_model`: the shallow text + CSS + CETR model from the paper
* `kohlschuetter_weninger_model`: includes the shallow text + CETR features

## A note about encoding

If you know the encoding of the document, you can pass it down to the parser:

    content = kohlschuetter_model.analyze(html_string, encoding='utf-8')

Otherwise, we try to guess the encoding from a `meta` tag or specified
`<?xml encoding=".."?>` tag.  If that fails, we assume "UTF-8".

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


# Details about the training data

A training data set consists of a collection of web pages and the extracted
"gold standard" content.  For our purposes we standardize  
a data set as a set of files on disk with a specific directory and naming
convention.  Each training example is all the data associated
with a single web page and all data for all examples lives under
a common `ROOTDIR`.

Each training example is identified by a common file root.
The data for example `X` lives in a set of sub-directories as follows:

* `$ROOTDIR/HTML/` contains the raw HTML named `X.html`
* `$ROOTDIR/Corrected/` contains the extracted content named `X.html.corrected.txt`

We will eventually provide a link to the training data.  Until then, if you would like
it send an e-mail to Matt Peters (address listed in our paper).

We have also tested our model on the data used in Weaning et al.
["CETR -- Content Extraction with Tag Ratios" (WWW 2010)](http://web.engr.illinois.edu/~weninge1/cetr/)
(scroll to the bottom for a link to their data).  We used the bash script
`cetr_to_dragnet.sh` to convert the data from CETR to Dragnet format.  In using their data,
we had to remove a small number of documents (less then 15) since they were so malformed
libxml2 could not parse them.  We also found some systematic problems with the data in the
`cleaneval-zh` and `myriad` data sets so decided not to use them.  For example,
many of the HTML files in `cleaneval-zh` contain several `</html>` tags, followed immediately
with `<DOCTYPE ..>` tags that libxml2 bonks out on.  Many of the gold standard files
in the `myriad` data contain significant portions of duplicated content that is not
present in the HTML document that we cannot use without a lot of manual cleanup.

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

    This solves the longest common sub-sequence problem to determine
    which blocks were extracted in the gold standard.
    Occasionally this will fail if lxml (libxml2) cannot parse
    a HTML document.  In this case, remove the offending document and restart the process.
2.  Run `split_data` to generate the `training.txt` and `test.txt` files split.
3.  Train the model with your selected features.  For example, to 
    train a model with the shallow text features from Kohlschuetter et al.
    and the CETR features from Weninger et al. use:

        from dragnet.model_training import train_models

        rootdir = '/my/data/directory/'
        output_prefix = '/path/to/output/kohlschuetter_weninger'
        features_to_use = ['kohlschuetter', 'weninger']
        train_models(rootdir, output_prefix, features_to_use, lam=100)

    This trains the model and writes a pickled version of it along with some
    some *block level* classification errors to a file.
    To compute the token level performance, see the next section.



## Evaluating content extraction models

Use `evaluate_models_tokens` in `model_training` to compute the token level
precision, recall and F1.  For example,
to evaluate the baseline model (keep everything) run:

    from dragnet.model_training import evaluate_models_tokens
    from dragnet.models import baseline_model

    rootdir = '/my/data/directory/'
    scores = evaluate_models_tokens(rootdir, baseline_model)


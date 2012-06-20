#! /usr/bin/env python

# Run a particular algorithm on the entire set of documents

from dragnet import Arias as technique

import os
for site in os.listdir('documents'):
    sitepath = os.path.join('documents', site)
    for document in os.listdir(sitepath):
        # Make sure the output directory exists
        try:
            os.makedirs(os.path.join('output', site))
        except:
            pass
        
        # Read in, analyze, write out
        inpath  = os.path.join(sitepath, document)
        outpath = os.path.join('output', site, document)
        
        print 'Working on %s' % inpath
        with open(inpath) as inf:
            with open(outpath, 'w+') as outf:
                outf.write(technique.analyze(inf.read(), inpath))

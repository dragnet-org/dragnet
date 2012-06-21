dragnet
=======
Dragnet isn't interested in the shiny chrome or boilerplate dressing of a 
webpage. It's interested in... 'just the facts.'

It is meant to become a collection of reference implementations of various 
dechroming / content extraction algorithms.

Each of the algorithms is implemented as a class of static methods that can be
imported from the top level of dragnet, and implement a method `analyze`, which 
accepts a string of HTML and returns a string representative of the content.

Running
-------
Fill a directory `documents` with per-site folders of the HTML sources of 
documents from that site, and then `run.py` will iterate through each of the 
input files and produce a corresponding file in `output` with just the content.
For example,

    documents/
        wired.com/
            latest-higgs-rumors
        seomoz.org/
            8-attributes-of-content-that-inspire-action

Arias et al.
------------
Based on [Language Independent Content Extraction from Web Pages](
    https://lirias.kuleuven.be/bitstream/123456789/215528/1/AriasEtAl2009.pdf)

    from dragnet import Arias
    import requests
    r = requests.get(
        'http://www.wired.com/wiredscience/2012/06/latest-higgs-rumors/')
    print Arias.analyze(r.content)

Kohlschütter et al.
-------------------
Based on [Boilerplate Detection using Shallow Text Features](
    http://www.l3s.de/~kohlschuetter/publications/wsdm187-kohlschuetter.pdf)
    
    from dragnet import Kohlschuetter
    import requests
    r = requests.get(
        'http://www.wired.com/wiredscience/2012/06/latest-higgs-rumors/')
    print Kohlschuetter.analyze(r.content)

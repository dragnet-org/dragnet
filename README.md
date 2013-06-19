dragnet
=======
Dragnet isn't interested in the shiny chrome or boilerplate dressing of a 
webpage. It's interested in... 'just the facts.'

Dragnet was published and presented at the 22nd International World Wide Web Conference
(WWW 2013).  See the paper
<a href="https://github.com/seomoz/dragnet/blob/master/dragnet_www2013.pdf">here.</a>

This repository is currently is an odd state where the lastest version of the code
used in the paper is on the `update_201211` branch and not on `master`.
If you are using Dragnet for a new project then we highly recommend using
that branch and not `master`.  Among other things, it contains the fully trained
models in the paper, whereas this `master` branch only contains the code but
not parameter files to run them.  We currently run the `master` branch in production
at Moz, but will merge in the `update_201211` branch once we have bandwidth
to test it fully at scale with our other back end systems.


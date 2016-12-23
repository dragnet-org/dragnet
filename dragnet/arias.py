#! /usr/bin/env python
"""
A *rough* implementation of that described by Aurias et al.:
https://lirias.kuleuven.be/bitstream/123456789/215528/1/AriasEtAl2009.pdf
"""
from .blocks import Blockifier
from .content_extraction_model import ContentExtractionModel, IdentityPredictor

import numpy as np
import scipy.weave


class AriasFeatures(object):
    """A global feature based on connected blocks of long text
      inspired by Arias"""
    nfeatures = 1

    def __init__(self, percent_cutoff, window):
        """Set parameters

        percent_cutoff = we use scipy.percentile(block_lengths, percent_cutoff)
            to determine the min length to call content
            percent_cutoff is a float in [0, 100]
        window = the window parameter to strip"""
        self._percent_cutoff = percent_cutoff
        self._window = window

    def __call__(self, blocks, train=False):
        from scipy import percentile
        features = np.zeros((len(blocks), AriasFeatures.nfeatures))

        block_lengths = np.array([len(block.text) for block in blocks])
        index = block_lengths.argmax()
        cutoff = int(percentile(block_lengths, self._percent_cutoff))
        lowindex, highindex = AriasFeatures.strip(block_lengths, index, self._window, cutoff)
        features[lowindex:(highindex + 1), 0] = 1.0
        return features

    @staticmethod
    def strip(block_lengths, index, window, cutoff):
        """Strip a list of blocks down to the content.

        Starting at some block index, expand outward to left and right until we
        encounter `window` consecutive blocks with length less then `cutoff`.

        block_lengths = 1D numpy array of length of text in blocks
            in document
        index = the starting index for the determination
        window = we need this many consecutive blocks <= cutoff to terminate
        cutoff = min block length to be considered content

        returns start/end block indices of the content
        """
        ret = np.zeros(2, np.int)
        nblock = len(block_lengths)
        c_code = """
            // First we'll work backwards to find the beginning index, and then we'll
            // work forward to find the ending index, and then we'll just take that
            // slice to be our content
            int lowindex  = index;
            int lastindex = index;
            while (lowindex > 0)
            {
                if (lastindex - lowindex > window)
                    break;
                if (block_lengths(lowindex) >= cutoff)
                    lastindex = lowindex;
                lowindex--;
            }
            ret(0) = lastindex;

            // Like above, except we're looking in the forward direction
            int highindex = index;
            lastindex = index;
            while (highindex < nblock)
            {
                if (highindex - lastindex > window)
                    break;
                if (block_lengths(highindex) >= cutoff)
                    lastindex = highindex;
                highindex++;
            }
            ret(1) = lastindex;
        """
        scipy.weave.inline(
            c_code,
            ['ret', 'nblock', 'index', 'window', 'cutoff', 'block_lengths'],
            type_converters=scipy.weave.converters.blitz)
        return ret


class Arias(ContentExtractionModel):
    def __init__(self, percentile_cutoff=25, window=3, blockifier=Blockifier, **kwargs):
        """A Arias model.

        percentile_cutoff, window = passed into AriasFeatures to determine the model
        blockifier = something that implements Blockify
        **kwags = any kwargs to pass into ContentExtractionModel.__init__"""
        features = [AriasFeatures(percentile_cutoff, window)]
        ContentExtractionModel.__init__(self, blockifier, features, IdentityPredictor, **kwargs)

    @staticmethod
    def plot(L, name, low, hi, cutoff):
        '''
        Helper method to plot the document (like in the paper)
        '''
        import numpy as np
        import matplotlib.pyplot as plt

        # First, plot up to the low point in blue...
        plt.bar(np.arange(low), [len(l) for l in L[0:low]], linewidth=0.0)
        # And now from low-high in red
        plt.bar(
            np.arange(low, hi + 1), [len(l) for l in L[low:hi + 1]],
            linewidth=0.0, color='r')
        # And from then on in blue
        plt.bar(
            np.arange(hi + 1, len(L)), [len(l) for l in L[hi + 1:]],
            linewidth=0.0)
        # Lastly, apply a line across the board at the cutoff
        plt.plot([0, len(L)], [cutoff, cutoff], 'g-')

        plt.xlabel('Order')
        plt.ylabel('Length')
        plt.title(name)
        plt.show()

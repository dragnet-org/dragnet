# -*- coding: utf-8 -*-
"""
A *rough* implementation of that described by Kohlschütter et al.:
   http://www.l3s.de/~kohlschuetter/publications/wsdm187-kohlschuetter.pdf
"""
import numpy as np

from .base import Feature
from ..util import sliding_window


class KohlschuetterFeatures(Feature):
    """
    The text density/link density features
    from Kohlschuetter. Implements the features interface.

    TODO: better docs
    """

    def fit(self, blocks, y=None):
        return self

    def transform(self, blocks, y=None):
        """
        Args:
            blocks (List[Block]): as output by :class:`Blockifier.blockify`
            y (None): This isn't used, it's only here for API consistency.

        Returns:
            `np.ndarray`: 2D array of shape (num blocks, 6), where values are
                floats corresponding to the link and text densities of
                a block and its immediate neighbors in the sequence.
        """
        nblocks = len(blocks)
        if nblocks < 3:
            raise ValueError(
                'at least 3 blocks are needed to make Kohlschuetter features')

        features = np.empty((nblocks, 6), dtype=float)
        i = 0
        features[i, :] = (
            0.0, 0.0,
            blocks[i].link_density, blocks[i].text_density,
            blocks[i + 1].link_density, blocks[i + 1].text_density
            )
        for i, (prevb, currb, nextb) in enumerate(sliding_window(blocks, 3)):
            features[i + 1, :] = (
                prevb.link_density, prevb.text_density,
                currb.link_density, currb.text_density,
                nextb.link_density, nextb.text_density
                )
        i = nblocks - 1
        features[i, :] = (
            blocks[i - 1].link_density, blocks[i - 1].text_density,
            blocks[i].link_density, blocks[i].text_density,
            0.0, 0.0
            )
        return features

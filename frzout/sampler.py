# -*- coding: utf-8 -*-

import numpy as np

from ._frzout import Surface, HRG, sample

__all__ = ['Sampler']


class Sampler(object):
    """
    Main sampler class.

    """
    def __init__(
            self, x, sigma, v, T, ymax=.5,
            species='all', res_width=True, decay_f500=False
    ):
        self._surface = Surface(
            np.asarray(x), np.asarray(sigma), np.asarray(v), ymax=ymax
        )
        self._hrg = HRG(
            T, species=species, res_width=res_width, decay_f500=decay_f500
        )

    @property
    def surface(self):
        """
        Internal Surface object.

        """
        return self._surface

    @property
    def hrg(self):
        """
        Internal HRG object.

        """
        return self._hrg

    @property
    def navg(self):
        """
        Average number of particles.

        """
        return self._surface.volume * self._hrg.density()

    def sample(self, unpack=False):
        """
        Sample once and return particle data, either as a numpy record array
        with fields (ID, x, p) or a tuple of unpacked arrays.

        """
        parts = sample(self._surface, self._hrg)
        if unpack:
            return parts['ID'], parts['x'], parts['p']
        else:
            return parts.view(np.recarray)

    def iter_samples(self, n, unpack=False):
        """
        Lazily iterate over samples.

        """
        return (self.sample(unpack=unpack) for _ in range(n))

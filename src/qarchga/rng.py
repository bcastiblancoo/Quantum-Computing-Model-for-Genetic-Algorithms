from dataclasses import dataclass
import numpy as np

@dataclass
class RNG:
    seed: int

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def integers(self, low, high=None, size=None):
        return self._rng.integers(low, high=high, size=size)

    def choice(self, a, size=None, replace=True, p=None):
        return self._rng.choice(a, size=size, replace=replace, p=p)

    def random(self, size=None):
        return self._rng.random(size=size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._rng.normal(loc=loc, scale=scale, size=size)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._rng.uniform(low, high, size=size)

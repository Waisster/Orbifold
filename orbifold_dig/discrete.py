import numpy as np

class DiscreteSet:
    def __init__(self, M, R):
        self.M = M
        self.R = R
        self.levels = np.linspace(-R, R, M)

    def random_level(self, rng):
        idx = rng.integers(0, self.M)
        return self.levels[idx], idx

    def neighbor_index(self, idx, rng):
        # choose left/right neighbor with wrap-around
        if self.M == 1:
            return idx
        step = -1 if rng.random() < 0.5 else 1
        return (idx + step) % self.M

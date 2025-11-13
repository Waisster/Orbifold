import numpy as np
from dataclasses import dataclass

@dataclass
class Lattice:
    Lt: int
    Lx: int
    Ly: int

    def shape_sites(self):
        return (self.Lt, self.Lx, self.Ly)

    def shift(self, arr, axis, step):
        return np.roll(arr, shift=step, axis=axis)

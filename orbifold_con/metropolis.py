import numpy as np
from copy import deepcopy
from .action import action_total

def local_update_matrix(mat, amplitude, rng):
    # Add a small random complex perturbation to a matrix
    perturb = amplitude * (rng.normal(size=mat.shape) + 1j * rng.normal(size=mat.shape))
    return mat + perturb

def metropolis_update(umat, zmat, lat, cfg, rng):
    accept = 0
    total = 0

    # Update time links (umat)
    it_max, ix_max, iy_max = umat.shape
    for it in range(it_max):
        for ix in range(ix_max):
            for iy in range(iy_max):
                old = umat[it, ix, iy]
                umat_new = deepcopy(umat)
                umat_new[it, ix, iy] = local_update_matrix(old, amplitude=0.10, rng=rng)  # Tune amplitude
                dS = action_total(umat_new, zmat, lat, cfg) - action_total(umat, zmat, lat, cfg)
                if np.log(rng.uniform()) < -dS:
                    umat[it, ix, iy] = umat_new[it, ix, iy]
                    accept += 1
                total += 1

    # Update space links (zmat)
    for j in range(2):
        it_max, ix_max, iy_max = zmat.shape[1:4]
        for it in range(it_max):
            for ix in range(ix_max):
                for iy in range(iy_max):
                    old = zmat[j, it, ix, iy]
                    zmat_new = deepcopy(zmat)
                    zmat_new[j, it, ix, iy] = local_update_matrix(old, amplitude=0.10, rng=rng)
                    dS = action_total(umat, zmat_new, lat, cfg) - action_total(umat, zmat, lat, cfg)
                    if np.log(rng.uniform()) < -dS:
                        zmat[j, it, ix, iy] = zmat_new[j, it, ix, iy]
                        accept += 1
                    total += 1

    acc_rate = accept / total
    return umat, zmat, acc_rate

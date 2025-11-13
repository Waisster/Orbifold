import numpy as np
from .discrete import DiscreteSet

def init_discrete_fields(cfg, rng):
    disc = DiscreteSet(cfg.M, cfg.R)
    # umat[t,x,y] : N x N complex
    umat = np.empty((cfg.Lt, cfg.Lx, cfg.Ly), dtype=object)
    umat_idx = np.empty((cfg.Lt, cfg.Lx, cfg.Ly), dtype=object)  # store indices per entry (real, imag)
    for it in range(cfg.Lt):
        for ix in range(cfg.Lx):
            for iy in range(cfg.Ly):
                U = np.zeros((cfg.N, cfg.N), dtype=complex)
                idx = np.zeros((cfg.N, cfg.N, 2), dtype=int)
                for a in range(cfg.N):
                    for b in range(cfg.N):
                        re, i_re = disc.random_level(rng)
                        im, i_im = disc.random_level(rng)
                        U[a, b] = re + 1j * im
                        idx[a, b, 0] = i_re
                        idx[a, b, 1] = i_im
                umat[it, ix, iy] = U
                umat_idx[it, ix, iy] = idx

    # zmat[j,t,x,y]
    zmat = np.empty((2, cfg.Lt, cfg.Lx, cfg.Ly), dtype=object)
    zmat_idx = np.empty((2, cfg.Lt, cfg.Lx, cfg.Ly), dtype=object)
    for j in range(2):
        for it in range(cfg.Lt):
            for ix in range(cfg.Lx):
                for iy in range(cfg.Ly):
                    Z = np.zeros((cfg.N, cfg.N), dtype=complex)
                    idx = np.zeros((cfg.N, cfg.N, 2), dtype=int)
                    for a in range(cfg.N):
                        for b in range(cfg.N):
                            re, i_re = disc.random_level(rng)
                            im, i_im = disc.random_level(rng)
                            Z[a, b] = re + 1j * im
                            idx[a, b, 0] = i_re
                            idx[a, b, 1] = i_im
                    zmat[j, it, ix, iy] = Z
                    zmat_idx[j, it, ix, iy] = idx
    return disc, umat, zmat, umat_idx, zmat_idx

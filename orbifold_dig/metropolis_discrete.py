import numpy as np
from copy import deepcopy
from .action_discrete import action_total

def propose_single_entry(matrix, idx_mat, a, b, which, disc, rng, neighbor_moves=True):
    # which: 0=real, 1=imag
    old_idx = idx_mat[a, b, which]
    if neighbor_moves:
        new_idx = disc.neighbor_index(old_idx, rng)
    else:
        new_idx = rng.integers(0, disc.M)
    if new_idx == old_idx and disc.M > 1:
        # ensure move
        new_idx = (old_idx + 1) % disc.M
    new_val = disc.levels[new_idx]
    return new_idx, new_val

def update_one_link_isolated(idx_tuple, cfg, disc, rng):
    # helper to pick a random link and entry
    is_umat = rng.random() < 0.5
    if is_umat:
        it = rng.integers(0, cfg.Lt); ix = rng.integers(0, cfg.Lx); iy = rng.integers(0, cfg.Ly)
        j = None
    else:
        j = rng.integers(0, 2)
        it = rng.integers(0, cfg.Lt); ix = rng.integers(0, cfg.Lx); iy = rng.integers(0, cfg.Ly)
    a = rng.integers(0, cfg.N); b = rng.integers(0, cfg.N)
    which = 0 if rng.random() < 0.5 else 1
    return is_umat, j, it, ix, iy, a, b, which

def metropolis_sweep(umat, zmat, umat_idx, zmat_idx, cfg, disc, rng):
    acc = 0
    tot = 0
    # one sweep ~ O(#dofs) local proposals
    n_proposals = cfg.Lt * cfg.Lx * cfg.Ly * cfg.N * cfg.N * 3  # rough count
    baseS = action_total(umat, zmat, cfg)

    for _ in range(n_proposals):
        is_umat, j, it, ix, iy, a, b, which = update_one_link_isolated(None, cfg, disc, rng)
        if is_umat:
            U_old = umat[it, ix, iy].copy()
            idx_old = umat_idx[it, ix, iy].copy()
            # propose new value
            new_idx, new_val = propose_single_entry(U_old, idx_old, a, b, which, disc, rng, cfg.neighbor_moves)
            U_new = U_old.copy()
            if which == 0:
                U_new[a, b] = new_val + 1j * U_new[a, b].imag
            else:
                U_new[a, b] = U_new[a, b].real + 1j * new_val
            umat[it, ix, iy] = U_new
            S_new = action_total(umat, zmat, cfg)
            dS = S_new - baseS
            if np.log(rng.random()) < -dS:
                baseS = S_new
                umat_idx[it, ix, iy][a, b, which] = new_idx
                acc += 1
            else:
                umat[it, ix, iy] = U_old
        else:
            Z_old = zmat[j, it, ix, iy].copy()
            idx_old = zmat_idx[j, it, ix, iy].copy()
            new_idx, new_val = propose_single_entry(Z_old, idx_old, a, b, which, disc, rng, cfg.neighbor_moves)
            Z_new = Z_old.copy()
            if which == 0:
                Z_new[a, b] = new_val + 1j * Z_new[a, b].imag
            else:
                Z_new[a, b] = Z_new[a, b].real + 1j * new_val
            zmat[j, it, ix, iy] = Z_new
            S_new = action_total(umat, zmat, cfg)
            dS = S_new - baseS
            if np.log(rng.random()) < -dS:
                baseS = S_new
                zmat_idx[j, it, ix, iy][a, b, which] = new_idx
                acc += 1
            else:
                zmat[j, it, ix, iy] = Z_old
        tot += 1
    return acc / max(1, tot)

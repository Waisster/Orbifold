import numpy as np
from action import local_action_U, local_action_Z
from config import z_from_indices, grid_values, grid_indices

rng = np.random.default_rng(seed=1234)

# random pertubation
# Constracting a traceless hermitian, expinential map it into SU(N)
def random_suN_perturb(eps, nmat, rng):
    # pure U(1): generate random phase e^(i * theta)
    theta = rng.normal(scale=eps)
    return np.exp(1j * theta)

# Update Temperal link U 
def metropolis_update_U(it, ix, iy, umat, zmat, beta, eps_U, at, as_, rng):
    S_old = local_action_U(it, ix, iy, umat, zmat, at, as_)
    U_old = umat[:, :, it, ix, iy].copy()
    dU = random_suN_perturb(eps_U, nmat=umat.shape[0], rng=rng)
    umat[:, :, it, ix, iy] = dU @ U_old
    S_new = local_action_U(it, ix, iy, umat, zmat, at, as_)
    dS = S_new - S_old
    if dS > 0.0 and rng.random() >= np.exp(-beta * dS):
        umat[:, :, it, ix, iy] = U_old
        return False
    return True

# update spatial link Z 
def metropolis_update_Z(it, ix, iy, d, umat, zmat, z_index, grid_values, grid_indicies, beta, at, as_, mass2, mass2_U1, rng, max_step=1):
    """
    Metropolis update for a single complex Z component at (it,ix,iy,d)
    in discrete coordinate basis:
      - z_index[0, a, b, it, ix, iy, d] : real index n_r
      - z_index[1, a, b, it, ix, iy, d] : imag index n_i
    Proposal: for each matrix element, n -> n + Δn with |Δn|<=max_step,
    staying within [-N_level, +N_level].
    """
    S_old = local_action_Z(it, ix, iy, d, umat, zmat, at, as_, mass2, mass2_U1)
    Z_old = zmat[:, :, it, ix, iy, d].copy()
    
    # backup old indices at this site
    real_old = z_index[0, :, :, it, ix, iy, d].copy()
    imag_old = z_index[1, :, :, it, ix, iy, d].copy()

    # propose new indices
    real_new = real_old.copy()
    imag_new = imag_old.copy()
    
    # loop over matrix entries a,b
    nmat = umat.shape[0]
    for a in range(nmat):
        for b in range(nmat):
            # proposal step in index space for real part
            delta_nr = rng.integers(-max_step, max_step + 1)
            new_nr = real_old[a, b] + delta_nr
            # clip to valid range [-N_level, N_level]
            new_nr = np.clip(new_nr,
                             grid_indices[0],
                             grid_indices[-1])
            real_new[a, b] = new_nr

            # proposal step in index space for imag part
            delta_ni = rng.integers(-max_step, max_step + 1)
            new_ni = imag_old[a, b] + delta_ni
            new_ni = np.clip(new_ni,
                             grid_indices[0],
                             grid_indices[-1])
            imag_new[a, b] = new_ni

    # write proposed indices into z_index
    z_index[0, :, :, it, ix, iy, d] = real_new
    z_index[1, :, :, it, ix, iy, d] = imag_new
    zmat = z_from_indices(z_index, grid_values)

    # new local action
    S_new = local_action_Z(it, ix, iy, d, umat, zmat, at, as_, mass2, mass2_U1)

    dS = S_new - S_old
    if dS > 0.0 and rng.random() >= np.exp(-beta * dS):
        # reject: revert indices
        z_index[0, :, :, it, ix, iy, d] = real_old
        z_index[1, :, :, it, ix, iy, d] = imag_old
        zmat[:, :, it, ix, iy, d] = Z_old
        return False
    return True
    
# run a sweep
def metropolis_sweep(umat, zmat, z_index, beta, eps_U, grid_indicies, grid_values, at, as_, mass2, mass2_U1, rng):
    nt, nx, ny = umat.shape[2:]
    accU = 0
    accZ = 0
    nU = nt * nx * ny
    nZ = 2 * nt * nx * ny

    for it in range(nt):
        for ix in range(nx):
            for iy in range(ny):
                if metropolis_update_U(it, ix, iy, umat, zmat, beta, eps_U, at, as_, rng):
                    accU += 1

    for d in range(2):
        for it in range(nt):
            for ix in range(nx):
                for iy in range(ny):
                    if metropolis_update_Z(it, ix, iy, d, umat, zmat, z_index, grid_values, 
                                           grid_indicies, beta, at, as_, mass2, mass2_U1, rng, max_step=1):
                        accZ += 1

    return accU / nU, accZ / nZ


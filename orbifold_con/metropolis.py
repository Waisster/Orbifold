import numpy as np
from action import local_action_U, local_action_Z

rng = np.random.default_rng(seed=1234)

# random pertubation
# Constracting a traceless hermitian, expinential map it into SU(N)
def random_suN_perturb(eps, nmat, rng):
    A = rng.normal(size=(nmat, nmat)) + 1j * rng.normal(size=(nmat, nmat))
    H = (A + A.conj().T) / 2.0
    H -= np.trace(H) * np.eye(nmat) / nmat
    w, v = np.linalg.eigh(1j * eps * H)
    U = v @ np.diag(np.exp(w)) @ v.conj().T
    return U

# add a random complex matrix
def random_complex_perturb(eps, shape, rng):
    eta = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return eps * eta

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
def metropolis_update_Z(it, ix, iy, d, umat, zmat, beta, eps_Z, at, as_, mass2, mass2_U1, rng):
    S_old = local_action_Z(it, ix, iy, d, umat, zmat, at, as_, mass2, mass2_U1)
    Z_old = zmat[:, :, it, ix, iy, d].copy()
    zmat[:, :, it, ix, iy, d] = Z_old + random_complex_perturb(eps_Z, Z_old.shape, rng)
    S_new = local_action_Z(it, ix, iy, d, umat, zmat, at, as_, mass2, mass2_U1)
    dS = S_new - S_old
    if dS > 0.0 and rng.random() >= np.exp(-beta * dS):
        zmat[:, :, it, ix, iy, d] = Z_old
        return False
    return True

# run a sweep
def metropolis_sweep(umat, zmat, beta, eps_U, eps_Z, at, as_, mass2, mass2_U1, rng):
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
                    if metropolis_update_Z(it, ix, iy, d, umat, zmat, beta, eps_Z,
                                           at, as_, mass2, mass2_U1, rng):
                        accZ += 1

    return accU / nU, accZ / nZ


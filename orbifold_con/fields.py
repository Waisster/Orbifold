import numpy as np

def random_suN(N, rng):
    X = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    Q, R = np.linalg.qr(X)
    # make det=1
    det = np.linalg.det(Q)
    Q *= det ** (-1.0 / N)
    return Q

def init_fields(cfg, lat, rng):
    # umat[t, x, y] in SU(N)
    umat = np.empty(lat.shape_sites(), dtype=object)
    for it in range(cfg.Lt):
        for ix in range(cfg.Lx):
            for iy in range(cfg.Ly):
                umat[it, ix, iy] = random_suN(cfg.N, rng) if cfg.init == 1 else np.eye(cfg.N, dtype=complex)
    # zmat[j, t, x, y] as complex matrices (j=0:x,1:y)
    zmat = np.empty((2,) + lat.shape_sites(), dtype=object)
    for j in range(2):
        for it in range(cfg.Lt):
            for ix in range(cfg.Lx):
                for iy in range(cfg.Ly):
                    # complex noncompact init
                    Z = (rng.normal(size=(cfg.N, cfg.N)) + 1j * rng.normal(size=(cfg.N, cfg.N))) / np.sqrt(2.0)
                    if cfg.init == 0:
                        Z = np.eye(cfg.N, dtype=complex)
                    zmat[j, it, ix, iy] = Z
    return umat, zmat

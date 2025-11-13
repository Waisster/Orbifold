import numpy as np

def s_mass(zmat, cfg):
    val = 0.0
    for j in range(2):
        for it in range(cfg.Lt):
            for ix in range(cfg.Lx):
                for iy in range(cfg.Ly):
                    Z = zmat[j, it, ix, iy]
                    H = Z.conj().T @ Z - np.eye(cfg.N, dtype=complex)
                    val += cfg.m2 * np.real(np.trace(H.conj().T @ H))
                    detZ = np.linalg.det(Z)
                    val += cfg.m2_u1 * (np.abs(detZ) - 1.0) ** 2
    return val

def s_kin(umat, zmat, cfg):
    val = 0.0
    for j in range(2):
        # forward in t
        for it in range(cfg.Lt):
            itp = (it + 1) % cfg.Lt
            for ix in range(cfg.Lx):
                ixp = (ix + (1 if j == 0 else 0)) % cfg.Lx
                for iy in range(cfg.Ly):
                    iyp = (iy + (1 if j == 1 else 0)) % cfg.Ly
                    Ut = umat[it, ix, iy]
                    term = Ut @ zmat[j, itp, ix, iy] - zmat[j, it, ix, iy] @ umat[it, ixp, iyp]
                    val += np.real(np.trace(term.conj().T @ term))
    return (cfg.beta * cfg.at / cfg.a) * val

def s_spa(zmat, cfg):
    val = 0.0
    for it in range(cfg.Lt):
        for ix in range(cfg.Lx):
            ixp = (ix + 1) % cfg.Lx
            for iy in range(cfg.Ly):
                iyp = (iy + 1) % cfg.Ly
                Zx = zmat[0, it, ix, iy]
                Zy = zmat[1, it, ix, iy]
                Zx_fx = zmat[0, it, ixp, iy]
                Zy_fy = zmat[1, it, ix, iyp]
                A = Zx @ Zy_fy
                B = Zy @ Zx_fx
                term = A - B
                val += np.real(np.trace(term.conj().T @ term))
    return (cfg.beta * cfg.a / cfg.at) * val

def action_total(umat, zmat, cfg):
    return s_kin(umat, zmat, cfg) + s_spa(zmat, cfg) + s_mass(zmat, cfg)

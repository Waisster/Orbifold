import numpy as np

def radial_penalty(Z):
    # Tr[(Z^\dagger Z - I)^2]
    N = Z.shape[0]
    H = Z.conj().T @ Z - np.eye(N, dtype=complex)
    return np.real(np.trace(H.conj().T @ H))

def u1_penalty(Z):
    # (|det U| - 1)^2 via polar unitary U ≈ Z (Z^\dagger Z)^{-1/2}
    # numerically, penalize |det(Z)| away from 1 as a proxy
    detZ = np.linalg.det(Z)
    return (np.abs(detZ) - 1.0) ** 2

def s_mass(zmat, cfg):
    val = 0.0
    for j in range(2):
        it_max, ix_max, iy_max = zmat.shape[1:4]
        for it in range(it_max):
            for ix in range(ix_max):
                for iy in range(iy_max):
                    Z = zmat[j, it, ix, iy]
                    val += cfg.m2 * radial_penalty(Z)
                    val += cfg.m2_u1 * u1_penalty(Z)
    return val

def s_kin(umat, zmat, lat, cfg):
    # || U_t(t,x,y) Z_j(t+1,x,y) - Z_j(t,x,y) U_t(t,x+e_j,y+e_j?) ||^2
    # Here we use a simple gauge-covariant nearest-neighbor difference in t
    val = 0.0
    for j in range(2):
        Z = zmat[j]
        Z_fwd_t = np.roll(Z, shift=-1, axis=1)  # forward in t
        it_max, ix_max, iy_max = Z.shape[1:4]
        for it in range(it_max):
            for ix in range(ix_max):
                for iy in range(iy_max):
                    Ut = umat[it, ix, iy]
                    term = Ut @ Z_fwd_t[j, it, ix, iy] - Z[j, it, ix, iy] @ Ut
                    val += np.real(np.trace(term.conj().T @ term))
    return (cfg.beta * cfg.at / cfg.a) * val

def s_spa(zmat, lat, cfg):
    # Plaquette-like: || Z_x Z_y(x+ex) - Z_y Z_x(x+ey) ||^2
    val = 0.0
    Zx = zmat[0]
    Zy = zmat[1]
    Zx_fwd_x = np.roll(Zx, shift=-1, axis=2)  # +x
    Zy_fwd_y = np.roll(Zy, shift=-1, axis=3)  # +y
    it_max, ix_max, iy_max = Zx.shape[1:4]
    for it in range(it_max):
        for ix in range(ix_max):
            for iy in range(iy_max):
                A = Zx[0, it, ix, iy] @ Zy_fwd_y[1, it, ix, iy]
                B = Zy[1, it, ix, iy] @ Zx_fwd_x[0, it, ix, iy]
                term = A - B
                val += np.real(np.trace(term.conj().T @ term))
    return (cfg.beta * cfg.a / cfg.at) * val

def action_total(umat, zmat, lat, cfg):
    return s_kin(umat, zmat, lat, cfg) + s_spa(zmat, lat, cfg) + s_mass(zmat, cfg)

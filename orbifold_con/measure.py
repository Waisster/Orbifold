import numpy as np

def polyakov_loop_abs(umat):
    Lt, Lx, Ly = umat.shape
    acc = 0.0
    for ix in range(Lx):
        for iy in range(Ly):
            P = np.eye(umat[0, ix, iy].shape[0], dtype=complex)
            for it in range(Lt):
                P = P @ umat[it, ix, iy]
            acc += np.abs(np.trace(P)) / P.shape[0]
    return acc / (Lx * Ly)

def radial_monitor(zmat):
    acc = 0.0
    count = 0
    for j in range(2):
        it_max, ix_max, iy_max = zmat.shape[1:4]
        for it in range(it_max):
            for ix in range(ix_max):
                for iy in range(iy_max):
                    Z = zmat[j, it, ix, iy]
                    N = Z.shape[0]
                    H = Z.conj().T @ Z - np.eye(N, dtype=complex)
                    acc += np.real(np.trace(H.conj().T @ H))
                    count += 1
    return acc / count

def u1_monitor(zmat):
    acc = 0.0
    count = 0
    for j in range(2):
        it_max, ix_max, iy_max = zmat.shape[1:4]
        for it in range(it_max):
            for ix in range(ix_max):
                for iy in range(iy_max):
                    Z = zmat[j, it, ix, iy]
                    acc += np.abs(np.linalg.det(Z))
                    count += 1
    # want -> 1 in Wilson limit
    return acc / count

def plaquette_like_energy(zmat):
    Zx = zmat[0]; Zy = zmat[1]
    Zx_f = np.roll(Zx, -1, axis=2)
    Zy_f = np.roll(Zy, -1, axis=3)
    it_max, ix_max, iy_max = Zx.shape[1:4]
    acc = 0.0
    for it in range(it_max):
        for ix in range(ix_max):
            for iy in range(iy_max):
                A = Zx[0, it, ix, iy] @ Zy_f[1, it, ix, iy]
                B = Zy[1, it, ix, iy] @ Zx_f[0, it, ix, iy]
                term = A - B
                acc += np.real(np.trace(term.conj().T @ term))
    return acc / (it_max * ix_max * iy_max)

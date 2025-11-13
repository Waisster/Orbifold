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
        for it in range(zmat.shape[1]):
            for ix in range(zmat.shape[2]):
                for iy in range(zmat.shape[3]):
                    Z = zmat[j, it, ix, iy]
                    N = Z.shape[0]
                    H = Z.conj().T @ Z - np.eye(N, dtype=complex)
                    acc += np.real(np.trace(H.conj().T @ H))
                    count += 1
    return acc / max(1, count)

def u1_monitor(zmat):
    acc = 0.0
    count = 0
    for j in range(2):
        for it in range(zmat.shape[1]):
            for ix in range(zmat.shape[2]):
                for iy in range(zmat.shape[3]):
                    Z = zmat[j, it, ix, iy]
                    acc += np.abs(np.linalg.det(Z))
                    count += 1
    return acc / max(1, count)

def plaquette_like_energy(zmat):
    acc = 0.0
    count = 0
    for it in range(zmat.shape[1]):
        for ix in range(zmat.shape[2]):
            ixp = (ix + 1) % zmat.shape[2]
            for iy in range(zmat.shape[3]):
                iyp = (iy + 1) % zmat.shape[3]
                Zx = zmat[0, it, ix, iy]
                Zy = zmat[1, it, ix, iy]
                Zx_fx = zmat[0, it, ixp, iy]
                Zy_fy = zmat[1, it, ix, iyp]
                term = Zx @ Zy_fy - Zy @ Zx_fx
                acc += np.real(np.trace(term.conj().T @ term))
                count += 1
    return acc / max(1, count)

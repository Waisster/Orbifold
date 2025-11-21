import numpy as np
from config import wrap

# spatial plaquette
def measure_plaquette_Z(zmat):
    """
    Using Zx, Zy constuct plaquette Tr(Zx Zy Zx^dagger Zy^dagger)。
    """
    nmat = zmat.shape[0]
    nt, nx, ny = zmat.shape[2:5]
    plaq = 0.0
    count = 0
    for it in range(nt):
        for ix in range(nx):
            ixp = wrap(ix+1, nx)
            for iy in range(ny):
                iyp = wrap(iy+1, ny)
                Zx = zmat[:, :, it, ix, iy, 0]
                Zy = zmat[:, :, it, ix, iy, 1]
                Zx_y = zmat[:, :, it, ix, iyp, 0]
                Zy_x = zmat[:, :, it, ixp, iy, 1]
                P = Zx @ Zy_x @ Zx_y.conj().T @ Zy.conj().T
                plaq += np.real(np.trace(P)) / nmat
                count += 1
    return plaq / count

# Temperal plaquette
def measure_plaquette_U(umat, zmat):
    """
    plaq_temp_U: temporal-spatial plaquette Tr(U Z U^dagger Z^dagger)。
    """
    nmat = umat.shape[0]
    nt, nx, ny = umat.shape[2:]
    plaq = 0.0
    count = 0
    for it in range(nt):
        itp = wrap(it+1, nt)
        for ix in range(nx):
            for iy in range(ny):
                U = umat[:, :, it, ix, iy]
                Zx = zmat[:, :, it, ix, iy, 0]
                U_tplus = umat[:, :, itp, ix, iy]
                Zx_tplus = zmat[:, :, itp, ix, iy, 0]
                P = U @ Zx_tplus @ U_tplus.conj().T @ Zx.conj().T
                plaq += np.real(np.trace(P)) / nmat
                count += 1
    return plaq / count

# Polyakov loop
def measure_polyakov(umat):
    """
    Polyakov loop: P(x,y) = Tr( Π_t U_t(t,x,y) ) / N
    """
    nmat = umat.shape[0]
    nt, nx, ny = umat.shape[2:]
    P_sum = 0.0 + 0.0j
    count = 0
    for ix in range(nx):
        for iy in range(ny):
            Uline = np.eye(nmat, dtype=np.complex128)
            for it in range(nt):
                Uline = umat[:, :, it, ix, iy] @ Uline
            P_site = np.trace(Uline) / nmat
            P_sum += P_site
            count += 1
    P_avg = P_sum / count
    return float(np.real(P_avg)), float(np.imag(P_avg))


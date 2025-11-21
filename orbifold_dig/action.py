import numpy as np
from config import wrap

# Local action S kin
def local_action_U(it, ix, iy, umat, zmat, at, as_):
    """
    only contains S_kin have umat(it,ix,iy) terms。
    Other term keep unchanged when we update fields for Metropolis

    """
    nmat = umat.shape[0]
    nt, nx, ny = umat.shape[2:]
    eye = np.eye(nmat, dtype=np.complex128)
    S = 0.0

    # 1) site (it,ix,iy) 
    itp = wrap(it+1, nt)
    ixp = wrap(ix+1, nx)
    iyp = wrap(iy+1, ny)

    U_here = umat[:, :, it, ix, iy]
    Zx_here = zmat[:, :, it, ix, iy, 0]
    Zy_here = zmat[:, :, it, ix, iy, 1]
    Zx_tplus = zmat[:, :, itp, ix, iy, 0]
    Zy_tplus = zmat[:, :, itp, ix, iy, 1]

    UZ_x = U_here @ Zx_tplus
    UZ_y = U_here @ Zy_tplus
    ZU_x = Zx_here @ umat[:, :, it, ixp, iy]
    ZU_y = Zy_here @ umat[:, :, it, ix, iyp]

    diff_x = UZ_x - ZU_x
    diff_y = UZ_y - ZU_y
    S += np.sum(np.abs(diff_x)**2) + np.sum(np.abs(diff_y)**2)

    # 2) Neighbour (it, ix-1, iy)
    ixm = wrap(ix-1, nx)
    Zx_left = zmat[:, :, it, ixm, iy, 0]
    ZU_x_left = Zx_left @ U_here
    UZ_x_left = umat[:, :, it, ixm, iy] @ zmat[:, :, itp, ixm, iy, 0]
    diff_left_x = UZ_x_left - ZU_x_left
    S += np.sum(np.abs(diff_left_x)**2)

    # 3) Neighbour (it, ix, iy-1) 
    iym = wrap(iy-1, ny)
    Zy_down = zmat[:, :, it, ix, iym, 1]
    ZU_y_down = Zy_down @ U_here
    UZ_y_down = umat[:, :, it, ix, iym] @ zmat[:, :, itp, ix, iym, 1]
    diff_down_y = UZ_y_down - ZU_y_down
    S += np.sum(np.abs(diff_down_y)**2)

    return S / at


def local_action_Z(it, ix, iy, d, umat, zmat, at, as_, mass2, mass2_U1):
    """
    zmat(:,:,it,ix,iy,d) involved terms：
    - kin: UZ/ZU
    - pot: commutator + laplacian-like
    - scalar mass
    - det constraint
    """
    nmat = umat.shape[0]
    nt, nx, ny = umat.shape[2:]
    eye = np.eye(nmat, dtype=np.complex128)

    S_kin = 0.0
    S_pot = 0.0
    S_mass = 0.0
    S_det = 0.0

    # Neighbours
    itp = wrap(it+1, nt)
    itm = wrap(it-1, nt)
    ixp = wrap(ix+1, nx)
    ixm = wrap(ix-1, nx)
    iyp = wrap(iy+1, ny)
    iym = wrap(iy-1, ny)

    Z_here = zmat[:, :, it, ix, iy, d]
    # 1) kin: Z on (it,ix,iy)
    if d == 0:  # Zx
        U_right = umat[:, :, it, ixp, iy]
        UZ = umat[:, :, it, ix, iy] @ zmat[:, :, itp, ix, iy, 0]
        ZU = Z_here @ U_right
        S_kin += np.sum(np.abs(UZ - ZU)**2) / at

       
    else:       # Zy
        U_up = umat[:, :, it, ix, iyp]
        UZ = umat[:, :, it, ix, iy] @ zmat[:, :, itp, ix, iy, 1]
        ZU = Z_here @ U_up
        S_kin += np.sum(np.abs(UZ - ZU)**2) / at

    # Z on t-1 neighbor 
    U_tm1 = umat[:, :, itm, ix, iy]
    Z_tm1 = Z_here
    if d == 0:
        ZU_tm1 = zmat[:, :, itm, ix, iy, 0] @ umat[:, :, itm, ixp, iy]
        UZ_tm1 = U_tm1 @ Z_tm1
        S_kin += np.sum(np.abs(UZ_tm1 - ZU_tm1)**2) / at
    else:
        ZU_tm1 = zmat[:, :, itm, ix, iy, 1] @ umat[:, :, itm, ix, iyp]
        UZ_tm1 = U_tm1 @ Z_tm1
        S_kin += np.sum(np.abs(UZ_tm1 - ZU_tm1)**2) / at

    # 2) pot: commutator plaquette like & laplacian-like
    # plaquette like
    Zx = zmat[:, :, it, ix, iy, 0]
    Zy = zmat[:, :, it, ix, iy, 1]
    Zy_xplus = zmat[:, :, it, ixp, iy, 1]
    Zx_yplus = zmat[:, :, it, ix, iyp, 0]
    ZxZy = Zx @ Zy_xplus
    ZyZx = Zy @ Zx_yplus
    comm = ZxZy - ZyZx
    S_pot += 2.0 * np.sum(np.abs(comm)**2) * at / (as_ * as_)
    # neighbour
    if d == 0:
        # Zx on y-1 neighbour
        Zx = zmat[:, :, it, ix, iym, 0]
        Zy = zmat[:, :, it, ix, iym, 1]
        Zy_xplus = zmat[:, :, it, ixp, iym, 1]
        Zx_yplus = zmat[:, :, it, ix, iy, 0]
        ZxZy = Zx @ Zy_xplus
        ZyZx = Zy @ Zx_yplus
        comm = ZxZy - ZyZx
        S_pot += 2.0 * np.sum(np.abs(comm)**2) * at / (as_ * as_)
    else:
        # Zy on x-1 neighbour
        Zx = zmat[:, :, it, ixm, iy, 0]
        Zy = zmat[:, :, it, ixm, iy, 1]
        Zy_xplus = zmat[:, :, it, ix, iy, 1]
        Zx_yplus = zmat[:, :, it, ixm, iyp, 0]
        ZxZy = Zx @ Zy_xplus
        ZyZx = Zy @ Zx_yplus
        comm = ZxZy - ZyZx
        S_pot += 2.0 * np.sum(np.abs(comm)**2) * at / (as_ * as_)

    # laplacian-like: Z Z† - Z†_{neighbor} Z_{neighbor}
    Zx_here = zmat[:, :, it, ix, iy, 0]
    ZxZxbar = Zx_here @ Zx_here.conj().T
    Zx_left = zmat[:, :, it, ixm, iy, 0]
    ZxbarZx_left = Zx_left.conj().T @ Zx_left
    lap_x = ZxZxbar - ZxbarZx_left

    Zy_here = zmat[:, :, it, ix, iy, 1]
    ZyZybar = Zy_here @ Zy_here.conj().T
    Zy_down = zmat[:, :, it, ix, iym, 1]
    ZybarZy_down = Zy_down.conj().T @ Zy_down
    lap_y = ZyZybar - ZybarZy_down
    S_pot += 0.5 * np.sum(np.abs(lap_x+lap_y)**2) * at / (as_ * as_)
   

    # 3) scalar mass at this site
    ZZbar = Z_here @ Z_here.conj().T
    Z_mass = ZZbar - 0.5 * eye
    S_mass += 0.5 * mass2 * at * np.sum(np.abs(Z_mass)**2)
    

    # 4) det constraint at this site
    if mass2_U1 != 0.0:
        c = 0.5 ** (-0.5 * float(nmat))
        Zloc = Z_here
        detZ = np.linalg.det(Zloc)
        temp = detZ * c - 1.0
        S_det += 0.5 * mass2_U1 * at * (np.abs(temp)**2)

    return float(S_kin + S_pot + S_mass + S_det)

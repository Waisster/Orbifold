import numpy as np

nmat = 1          # SU(N) with N = nmat
nt, nx, ny = 8, 8, 8
ndim = 3          # 1 time + 2 space

at = 0.3    # temporal lattice spacing
as_ = 0.3   # spatial lattice spacing
mass2 = 300.0   # mass square
mass2_U1 = 300.0    # U1 mass square
beta = 1.0        # effective 1/g^2 absorbed in S

rng = np.random.default_rng(seed=1234)  # NumPy RNG

# umat: temporal link; zmat: complex scalars (x,y directions)
umat = np.zeros((nmat, nmat, nt, nx, ny), dtype=np.complex128)
z_index = np.zeros((2, nmat, nmat, nt, nx, ny, ndim - 1), dtype=int)

# Wrap function for boundery condition
def wrap(i, L):
    return (i + L) % L

# Tuncation
# x_n = n * delta_x, n = -N,...,+N
N_level = 20            # truncation level: 2*N_level+1 points
delta_x = 0.5
grid_indices = np.arange(-N_level, N_level + 1, dtype=int)
grid_values = grid_indices * delta_x   # 1D real grid

def z_from_indices(z_index, grid_values):
    """
    Build complex Z field from index representation.
    z_index: (2, nmat, nmat, nt, nx, ny, 2)
    Return:
        zmat: (nmat, nmat, nt, nx, ny, 2), complex128
    """
    real_idx = z_index[0]
    imag_idx = z_index[1]
    # Broadcast grid_values over index arrays
    Z_real = grid_values[real_idx]
    Z_imag = grid_values[imag_idx]
    return Z_real + 1j * Z_imag

# Cold start
def cold_start(umat, z_index, as_, ndim):
    eye = np.eye(umat.shape[0], dtype=np.complex128)
    nt, nx, ny = umat.shape[2:]

    # choose some base value in the grid sqrt(1/2)
    base_val = np.sqrt(as_**(float(ndim - 3)) / 2.0)  # ndim=3 -> sqrt(1/2)
    # find closest index for base_val
    base_idx = int(np.argmin(np.abs(grid_values - base_val)))

    for it in range(nt):
        for ix in range(nx):
            for iy in range(ny):
                umat[:, :, it, ix, iy] = eye
                for d in range(ndim - 1):
                    # all real/imag parts at this site set to base_idx
                    z_index[0, :, :, it, ix, iy, d] = base_idx  # real
                    z_index[1, :, :, it, ix, iy, d] = base_idx  # imag


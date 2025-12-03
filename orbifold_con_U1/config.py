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
zmat = np.zeros((nmat, nmat, nt, nx, ny, ndim-1), dtype=np.complex128)

# Wrap function for boundery condition
def wrap(i, L):
    return (i + L) % L

# Cold start
def cold_start(umat, zmat, at, as_, ndim):
    eye = np.eye(umat.shape[0], dtype=np.complex128)
    nt, nx, ny = umat.shape[2:]
    for it in range(nt):
        for ix in range(nx):
            for iy in range(ny):
                umat[:, :, it, ix, iy] = eye
                factor = np.sqrt(as_**(float(ndim - 3)) / 2.0)  # ndim=3 -> sqrt(1/2)
                for d in range(ndim - 1):
                    zmat[:, :, it, ix, iy, d] = eye * factor


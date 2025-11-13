from dataclasses import dataclass

@dataclass
class Config:
    # Lattice sizes
    Lt: int = 4
    Lx: int = 8
    Ly: int = 8
    # Gauge group SU(N)
    N: int = 3
    # Couplings and anisotropy
    beta: float = 6.0
    at: float = 1.0
    a: float = 1.0
    # Orbifold mass terms
    m2: float = 50.0            # radial penalty
    m2_u1: float = 50.0         # U(1) penalty
    # HMC
    ntau: int = 10
    dtau_t: float = 0.02        # time-link step
    dtau_s: float = 0.02        # space-link step
    n_therm: int = 100
    n_traj: int = 1000
    nskip: int = 10
    # Init and RNG
    init: int = 1               # 1=random, 0=unit
    seed: int = 1234
    # Logging
    log_interval: int = 10
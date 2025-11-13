from dataclasses import dataclass

@dataclass
class Config:
    Lt: int = 4
    Lx: int = 6
    Ly: int = 6
    N: int = 2
    beta: float = 4.0
    at: float = 1.0
    a: float = 1.0
    m2: float = 100.0
    m2_u1: float = 100.0
    n_therm: int = 200
    n_steps: int = 2000
    nskip: int = 20
    log_interval: int = 20
    seed: int = 123

    # coordinate-basis truncation
    M: int = 8             # number of discrete levels per real/imag part
    R: float = 1.0         # range [-R, R]
    neighbor_moves: bool = True  # if True, propose nearest-neighbor level; else random level

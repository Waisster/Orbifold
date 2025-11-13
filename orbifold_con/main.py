import numpy as np
from src.config import Config
from src.lattice import Lattice
from src.fields import init_fields
from src.metropolis import metropolis_update
from src.measure import polyakov_loop_abs, radial_monitor, u1_monitor, plaquette_like_energy

def run():
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)
    lat = Lattice(cfg.Lt, cfg.Lx, cfg.Ly)
    umat, zmat = init_fields(cfg, lat, rng)

    # 热化
    n_therm_steps = cfg.n_therm
    for step in range(n_therm_steps):
        umat, zmat, acc_rate = metropolis_update(umat, zmat, lat, cfg, rng)

    for traj in range(1, cfg.n_traj + 1):
        umat, zmat, acc_rate = metropolis_update(umat, zmat, lat, cfg, rng)
        if traj % cfg.log_interval == 0:
            print(f"[step={traj}] acc={acc_rate:.2f}")

        if traj % cfg.nskip == 0:
            P = polyakov_loop_abs(umat)
            R = radial_monitor(zmat)
            U1 = u1_monitor(zmat)
            Epl = plaquette_like_energy(zmat)
            print(f"meas: |P|={P:.4f}  radial={R:.4e}  u1={U1:.4f}  Epl={Epl:.4e}")

if __name__ == "__main__":
    run()

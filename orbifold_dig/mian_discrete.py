import numpy as np
from src.config import Config
from src.fields_discrete import init_discrete_fields
from src.metropolis_discrete import metropolis_sweep
from src.action_discrete import action_total
from src.measure import polyakov_loop_abs, radial_monitor, u1_monitor, plaquette_like_energy

def run():
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)
    disc, umat, zmat, umat_idx, zmat_idx = init_discrete_fields(cfg, rng)

    # thermalization
    for s in range(cfg.n_therm):
        acc = metropolis_sweep(umat, zmat, umat_idx, zmat_idx, cfg, disc, rng)

    for s in range(1, cfg.n_steps + 1):
        acc = metropolis_sweep(umat, zmat, umat_idx, zmat_idx, cfg, disc, rng)
        if s % cfg.log_interval == 0:
            print(f"[step={s}] acc={acc:.2f}")
        if s % cfg.nskip == 0:
            P = polyakov_loop_abs(umat)
            R = radial_monitor(zmat)
            U1 = u1_monitor(zmat)
            Epl = plaquette_like_energy(zmat)
            S = action_total(umat, zmat, cfg)
            print(f"meas: |P|={P:.4f}  radial={R:.3e}  u1={U1:.4f}  Epl={Epl:.3e}  S={S:.3e}")

if __name__ == "__main__":
    run()

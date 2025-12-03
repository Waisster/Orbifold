from config import cold_start, at, as_, ndim, beta, mass2, mass2_U1, rng, umat, zmat
from metropolis import metropolis_sweep
from measure import measure_plaquette_U, measure_plaquette_Z, measure_polyakov
import pandas as pd
from tqdm import tqdm
import time

def run_monte_carlo(
    umat, zmat,
    n_therm=5000, n_meas=5000, meas_interval=10,
    eps_U=0.1, eps_Z=0.1
):
    cold_start(umat, zmat, at, as_, ndim)

    # Heating
    print("Thermalization...")
    for i in tqdm(range(n_therm)):
        aU, aZ = metropolis_sweep(umat, zmat, beta, eps_U, eps_Z,
                                  at, as_, mass2, mass2_U1, rng)

    # test
    print("Measurement...")
    results = []
    start_time = time.time()
    for istep in tqdm(range(n_meas)):
        aU, aZ = metropolis_sweep(umat, zmat, beta, eps_U, eps_Z,
                                  at, as_, mass2, mass2_U1, rng)
        if (istep + 1) % meas_interval == 0:
            # S_tot = total_action(umat, zmat, at, as_, mass2, mass2_U1)
            plaq_Z = measure_plaquette_Z(zmat)
            plaq_U = measure_plaquette_U(umat, zmat)
            P_re, P_im = measure_polyakov(umat)
            results.append({
                "step": istep + 1,
                # "S": S_tot,
                "plaq_Z": plaq_Z,
                "plaq_U": plaq_U,
                "Pol_re": P_re,
                "Pol_im": P_im,
                "accU": aU,
                "accZ": aZ,
            })
    return results

# save
import csv

def save_results_csv(filename, results):
    keys = ["step", "S", "plaq_Z", "plaq_U", "Pol_re", "Pol_im", "accU", "accZ"]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

if __name__ == "__main__":
    results = run_monte_carlo(umat, zmat,
                              n_therm=10000,
                              n_meas=5000,
                              meas_interval=10,
                              eps_U=0.1,
                              eps_Z=0.05)
    save_results_csv("orbifold_metropolis_con_SU2_8x8x8.csv", results)

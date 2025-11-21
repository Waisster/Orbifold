from config import cold_start, z_from_indices, at, as_, ndim, beta, mass2, mass2_U1, rng, umat, z_index, grid_indices, grid_values
from metropolis import metropolis_sweep
from measure import measure_plaquette_U, measure_plaquette_Z, measure_polyakov
import pandas as pd

def run_monte_carlo(
    umat, z_index, grid_indices,
    n_therm=5000, n_meas=5000, meas_interval=10,
    eps_U=0.1
):
    cold_start(umat, z_index, as_, ndim)
    zmat = z_from_indices(z_index, grid_values)

    # Heating
    for i in range(n_therm):
        aU, aZ = metropolis_sweep(umat, zmat, z_index, beta, eps_U, 
                                  grid_indices, grid_values, at, as_, mass2, mass2_U1, rng)

    # test
    results = []
    for istep in range(n_meas):
        aU, aZ = metropolis_sweep(umat, zmat, z_index, beta, eps_U, 
                                  grid_indices, grid_values, at, as_, mass2, mass2_U1, rng)
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
    results = run_monte_carlo(umat, z_index, grid_indices,
                              n_therm=10000,
                              n_meas=5000,
                              meas_interval=10,
                              eps_U=0.1,
                              )
    save_results_csv("orbifold_metropolis_dig_SU2_8x8x8.csv", results)

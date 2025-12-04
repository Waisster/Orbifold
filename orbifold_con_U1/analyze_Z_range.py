from config import cold_start, at, as_, ndim, beta, mass2, mass2_U1, rng, umat, zmat
from metropolis import metropolis_sweep
from tqdm import tqdm
import time
import numpy as np
import csv

def run_mc_continuous(umat, zmat, num_configs=20,
    n_therm=5000, eps_U=0.1, eps_Z=0.1):
    """
    Run a short continuous-Z Monte Carlo and return a list of zmat configurations.
    Each zmat has shape (nmat, nmat, nt, nx, ny, 2).
    """
    zmat_list = []
    # init
    cold_start(umat, zmat, at, as_, ndim) 

    # Heating
    print("Thermalization...")
    for i in tqdm(range(n_therm)):
        aU, aZ = metropolis_sweep(umat, zmat, beta, eps_U, eps_Z,
                                  at, as_, mass2, mass2_U1, rng)

    # sampling num_configs
    for iconf in range(num_configs):
        print(f"Config {iconf}")
        for _ in tqdm(range(100)):  # every 100 steps
            aU, aZ = metropolis_sweep(umat, zmat, beta, eps_U, eps_Z,
                                  at, as_, mass2, mass2_U1, rng)
        # save copy
        zmat_list.append(zmat.copy())
    return zmat_list



def analyze_Z_range(zmat_list):
    """
    zmat_list: list of arrays with shape (nmat,nmat,nt,nx,ny,2)
    """
    vals = []
    vals_re = []
    vals_im = []
    for zmat in zmat_list:
        vals_re.append(zmat.real.ravel())
        vals_im.append(zmat.imag.ravel())
    vals_re = np.concatenate(vals_re)
    vals_im = np.concatenate(vals_im)
    # mean = np.mean(vals)
    # std = np.std(vals)
    # p99 = np.quantile(np.abs(vals), 0.99)
    # max_abs = np.max(np.abs(vals))
    # print(f"mean = {mean:.3g}")
    # print(f"std  = {std:.3g}")
    # print(f"p99(|Z|) = {p99:.3g}")
    # print(f"max(|Z|) = {max_abs:.3g}")
    return vals_re, vals_im

if __name__ == "__main__":
    zmat_cont_list = run_mc_continuous(umat, zmat, num_configs=20,
    n_therm=1000, eps_U=0.1, eps_Z=0.1)
    vals_re, vals_im= analyze_Z_range(zmat_cont_list)

    keys = ["ReZ", "ImZ"]
    with open("Zij_distribution.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r, im in zip(vals_im, vals_re):
            writer.writerow({"ReZ": float(r), "ImZ": float(im)})


    # 根据输出选截断参数
    # R = max(4 * std, 1.2 * p99)
    # print(f"Suggest R ≈ {R:.3g}")

    # 如果你想要 2*N_level+1 = 15 个格点：
    # N_level = 7
    # delta_x = R / N_level
    # print(f"With N_level={N_level}, choose delta_x ≈ {delta_x:.3g}")

    # plot
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(5,4))
    # plt.hist(vals_re, bins=40, density=True, alpha=0.8, color="tab:blue")
    # plt.xlabel(r"$Z_{ij}$")
    # plt.ylabel("Probability density")
    # plt.title(r"Distribution of matrix elements $Z_{ij}$")
    # plt.tight_layout()
    # plt.savefig("Z_distribution_continuous.png", dpi=300)
    # plt.close()
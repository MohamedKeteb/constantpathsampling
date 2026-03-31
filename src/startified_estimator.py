import numpy as np
import matplotlib.pyplot as plt
from src.debiasedalgo import *



#---------------- Stratified estimator construction (splitting bins)----------------


def _build_bin(a, b, estimator, n_samples):
    lam = np.random.uniform(a, b, n_samples)
    y = np.array([estimator(l) for l in lam])
    return {"interval": (a, b), "lam": lam, "y": y}


def _split_bin(bin_data, estimator, n_samples):
    a, b = bin_data["interval"]
    mid = 0.5 * (a + b)

    lam = bin_data["lam"]
    y = bin_data["y"]
    left_mask = lam < mid

    left_lam = lam[left_mask]
    left_y = y[left_mask]
    right_lam = lam[~left_mask]
    right_y = y[~left_mask]

    missing_left = max(0, n_samples - len(left_lam))
    if missing_left > 0:
        new_left_lam = np.random.uniform(a, mid, missing_left)
        new_left_y = np.array([estimator(l) for l in new_left_lam])
        left_lam = np.concatenate((left_lam, new_left_lam))
        left_y = np.concatenate((left_y, new_left_y))

    missing_right = max(0, n_samples - len(right_lam))
    if missing_right > 0:
        new_right_lam = np.random.uniform(mid, b, missing_right)
        new_right_y = np.array([estimator(l) for l in new_right_lam])
        right_lam = np.concatenate((right_lam, new_right_lam))
        right_y = np.concatenate((right_y, new_right_y))

    left_bin = {"interval": (a, mid), "lam": left_lam, "y": left_y}
    right_bin = {"interval": (mid, b), "lam": right_lam, "y": right_y}
    return left_bin, right_bin


def splitting_grid(L, estimator, initial_L, n_samples):
    """
    Construire une grid greedy à partir d'une grid uniform initiale.
    
    Parameters:
    - L : nombre final de bins
    - estimator : estimateur à utiliser pour calculer la variance dans chaque bin
    - initial_bins : nombre initial de bins uniformes
    - n_samples : nombre cible de particules par bin
    """
    # Créer une grid initiale uniforme
    uniform_grid = np.linspace(0.0, 1.0, initial_L + 1)
    bins = [
        _build_bin(uniform_grid[i], uniform_grid[i + 1], estimator, n_samples)
        for i in range(initial_L)
    ]
    
    # Greedy split jusqu'à obtenir L bins
    while len(bins) < L:
        # Calculer la variance de chaque bin
        vars_ = [np.var(bin_data["y"]) for bin_data in bins]
        # Choisir le bin avec variance maximale
        idx = np.argmax(vars_)
        left_bin, right_bin = _split_bin(bins[idx], estimator, n_samples)

        # Remplacer le bin par ses deux moitiés en complétant seulement si besoin
        bins.pop(idx)
        bins.insert(idx, left_bin)
        bins.insert(idx + 1, right_bin)
    
    # Construire la grid finale
    grid = [bins[0]["interval"][0]]  # commencer par la borne inférieure du premier bin
    for bin_data in bins:
        _, b = bin_data["interval"]
        grid.append(b)
    return np.array(grid)

def stratified_estimator(grid, estimator, n_per_bin = 1):
    L = len(grid)-1
    start_est = 0.0
    for l in range(L):
        a, b = grid[l], grid[l+1]
        lam = np.random.uniform(a, b, n_per_bin)
        y = np.array([estimator(l) for l in lam])
        start_est += (b-a) * np.mean(y)
    return start_est


def simulate_stratified_estimator(grid, estimator, n_per_bin=1, nrep=100, ci_level=0.95, plot=True):
    """
    Simule plusieurs réplications de l'estimateur stratifié et trace
    un boxplot ainsi qu'un intervalle de confiance pour sa moyenne.

    Parameters:
    - grid : grille de stratification
    - estimator : fonction à intégrer
    - n_per_bin : nombre de particules par bin
    - nrep : nombre de réplications Monte Carlo
    - ci_level : niveau de confiance de l'intervalle
    - plot : si True, affiche les graphiques
    """
    estimates = np.array([
        stratified_estimator(grid, estimator, n_per_bin=n_per_bin)
        for _ in range(nrep)
    ])

    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1) if nrep > 1 else 0.0
    alpha = 1.0 - ci_level
    z_value = 1.959963984540054
    if ci_level != 0.95:
        from statistics import NormalDist
        z_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    margin = z_value * std_estimate / np.sqrt(nrep) if nrep > 1 else 0.0
    ci_low = mean_estimate - margin
    ci_high = mean_estimate + margin

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)

        axes[0].boxplot(estimates, vert=True, patch_artist=True)
        axes[0].set_title("Boxplot of stratified estimates")
        axes[0].set_ylabel("Estimate")
        axes[0].grid(True, axis="y", alpha=0.3)

        axes[1].errorbar(
            [0],
            [mean_estimate],
            yerr=[[mean_estimate - ci_low], [ci_high - mean_estimate]],
            fmt="o",
            capsize=6,
        )
        axes[1].set_title(f"{int(100 * ci_level)}% confidence interval")
        axes[1].set_ylabel("Estimate")
        axes[1].set_xticks([0])
        axes[1].set_xticklabels(["Mean"])
        axes[1].grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

    return {
        "estimates": estimates,
        "mean": mean_estimate,
        "std": std_estimate,
        "ci": (ci_low, ci_high),
    }, print("Mean:", mean_estimate, "CI:", (ci_low, ci_high))



#-- Startification with fix budget ----------------


def build_M_budget(L, total_budget):
    pass 

def build_grid_for_budget(L, M_budget):
    """
    Construit une grille sur [0, 1] avec L bins uniformes de taille 1/L,
    puis raffine chaque bin en sous-intervalles uniformes.

    Parameters:
    - L : nombre de bins de la grille de base
    - M_budget : vecteur de taille L donnant le nombre de
      sous-intervalles dans chaque bin
    """
    M_budget = np.asarray(M_budget, dtype=int)

    if M_budget.ndim != 1 or M_budget.shape[0] != L:
        raise ValueError("M_budget must be a vector of length L.")
    if np.any(M_budget < 1):
        raise ValueError("All entries of M_budget must be positive.")

    grid = [0.0]
    for i in range(L):
        a = i / L
        b = (i + 1) / L
        local_grid = np.linspace(a, b, M_budget[i] + 1)
        grid.extend(local_grid[1:])

    return np.array(grid)








def stratified_estimator_indep(L, M_budget, estimator):
    M_budget = np.array(M_budget)
    assert M_budget.size == 1 and np.sum(M_budget) >= L and M_budget.shape[0] == L, "M_budget should be a scalar or insufficient"
    uniform_grid = np.arange(0, L + 1) / L












#-- startification with (budget allocation with common random number within bins)  ----------------








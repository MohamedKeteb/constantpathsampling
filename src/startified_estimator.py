import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.debiasedalgo import *


#---------------- Stratified estimator construction given  grid ----------------

def stratified_estimator(grid, estimator, n_per_bin):
    L = len(grid)-1
    start_est = 0.0
    for l in range(L):
        a, b = grid[l], grid[l+1]
        lam = np.random.uniform(a, b, n_per_bin)
        y = np.array([estimator(l) for l in lam])
        start_est += (b-a) * np.mean(y)
    return start_est


#---------------- Stratified estimator construction by splitting----------------


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



def stratified_estimator_splitting(L, estimator, initial_L, n_samples):
    grid = splitting_grid(L, estimator, initial_L, n_samples)
    return stratified_estimator(grid, estimator, n_per_bin=1)


  
#--  Startification with budget and independent within bins----------------



def build_M_budget(L, total_budget, estimator, n_samples_per_bin=10):
    budegt_array = np.zeros(L, dtype=int)
    for i in range(L):
        a = i / L
        b = (i + 1) / L
        lam = np.random.uniform(a, b, n_samples_per_bin)
        y = np.array([estimator(l) for l in lam])
        var_estimate = np.var(y)
        budegt_array[i] = max(1, int(np.floor(var_estimate * total_budget / np.sum(var_estimate))))
    return budegt_array


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



def stratified_estimator_indepdent(L, M_budget, estimator):
    grid = build_grid_for_budget(L, M_budget)
    return stratified_estimator(grid, estimator, n_per_bin=1)



#-- startification with common random number  ----------------



def stratified_estimator_crn(L, M_budget, estimator):
    grid = build_grid_for_budget(L, M_budget)

    # Build the list of sub-bins contained in each main bin.
    subbins_by_bin = []
    start = 0
    for i in range(L):
        n_subbins = int(M_budget[i])
        local_subbins = []
        for j in range(n_subbins):
            local_subbins.append((grid[start + j], grid[start + j + 1]))
        subbins_by_bin.append(local_subbins)
        start += n_subbins

    strat_est = 0.0

    # Use one common uniform draw per sub-bin index and map it across bins.
    max_subbins = int(np.max(M_budget))
    for j in range(max_subbins):
        u = np.random.uniform()

        values = []
        widths = []
        for i in range(L):
            if j < M_budget[i]:
                a, b = subbins_by_bin[i][j]

                # Reuse the same random variable u in every bin by affine scaling.
                lam = a + (b - a) * u
                values.append(estimator(lam))
                widths.append(b - a)

        for width, value in zip(widths, values):
            strat_est += width * value

    return strat_est

#---------------- Stratified estimator with proportional allocation ---------------
def propo_startified_estimator(alpha, M, estimator):
    assert 0 < alpha < 1, "alpha must be in (0, 1)"

    budegt_startum = np.floor(M ** (1 - alpha))
    L = int(np.floor(M ** alpha))
    grid = np.linspace(0, 1, L + 1)
    return stratified_estimator(grid, estimator, n_per_bin=int(budegt_startum))

#---------------- Stratified estimator with optimal allocation ---------------

def stratified_estimator_bis(L, estimator, budget):
    grid = np.linspace(0, 1, L + 1)
    start_est = 0.0
    for l in range(L):
        a, b = grid[l], grid[l+1]
        lam = np.random.uniform(a, b, int(budget[l]))
        y = np.array([estimator(l) for l in lam])
        start_est += (b-a) * np.mean(y)
    return start_est

def optimal_startified_estimator(alpha, M, estimator):
    assert 0 < alpha < 1, "alpha must be in (0, 1)"
    L = int(np.floor(M ** alpha))
    budget = build_M_budget(L, M, estimator, n_samples_per_bin=10)
    return stratified_estimator_bis(L, estimator, budget)











#--------------------Simulation box plot & confidence interval----------------




def plot_estimators_box_CI(
    estimators,
    labels,
    nrep=100,
    ci_level=0.95,
    figsize=(12, 5),
):
    """
    Compare a family of stratified estimators with boxplots and confidence intervals.

    Parameters:
    - estimators: list of callables with no argument, each returning one estimate
    - labels: list of names for the estimators
    - nrep: number of Monte Carlo replications
    - ci_level: confidence level for the mean confidence intervals
    - figsize: figure size
    """
    if len(estimators) != len(labels):
        raise ValueError("estimators and labels must have the same length.")

    alpha = 1.0 - ci_level
    z_value = 1.959963984540054
    if ci_level != 0.95:
        from statistics import NormalDist
        z_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    samples = []
    means = []
    ci_lows = []
    ci_highs = []
    ci_lengths = []

    for estimator in estimators:
        values = np.array([estimator() for _ in range(nrep)])
        mean_value = np.mean(values)
        std_value = np.std(values, ddof=1) if nrep > 1 else 0.0
        margin = z_value * std_value / np.sqrt(nrep) if nrep > 1 else 0.0

        ci_low = mean_value - margin
        ci_high = mean_value + margin

        samples.append(values)
        means.append(mean_value)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)
        ci_lengths.append(ci_high - ci_low)

    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#4D4D4D",
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.8,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

    axes[0].boxplot(
        samples,
        tick_labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor="#DCEAF4", edgecolor="#4D4D4D"),
        medianprops=dict(color="#D55E00", linewidth=2),
        whiskerprops=dict(color="#4D4D4D"),
        capprops=dict(color="#4D4D4D"),
        flierprops=dict(markerfacecolor="#0072B2", markeredgecolor="#0072B2", markersize=4, alpha=0.6),
    )
    axes[0].set_title("Distribution of the estimators")
    axes[0].set_ylabel("Estimate")
    axes[0].grid(True, axis="y")
    axes[0].grid(False, axis="x")

    x = np.arange(len(labels))
    means = np.array(means)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)
    ci_lengths = np.array(ci_lengths)

    axes[1].errorbar(
        x,
        means,
        yerr=[means - ci_lows, ci_highs - means],
        fmt="o",
        color="#0072B2",
        ecolor="#0072B2",
        elinewidth=2,
        capsize=6,
        markersize=7,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title(f"Mean estimates with {int(100 * ci_level)}% confidence intervals")
    axes[1].set_ylabel("Estimate")
    axes[1].grid(True, axis="y")
    axes[1].grid(False, axis="x")

    axes[2].bar(x, ci_lengths, color="#D55E00", edgecolor="#4D4D4D", width=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_title("Confidence interval lengths")
    axes[2].set_ylabel("CI length")
    axes[2].grid(True, axis="y")
    axes[2].grid(False, axis="x")

    plt.tight_layout()
    plt.show()

    for label, ci_len in zip(labels, ci_lengths):
        print(f"{label}: CI length = {ci_len:.6f}")

    return {
        "samples": samples,
        "means": means,
        "ci_lows": ci_lows,
        "ci_highs": ci_highs,
        "ci_lengths": ci_lengths,
    }



def plot_estimators_histograms(
    estimators,
    labels,
    nrep=200,
    bins=30,
    figsize=(8, 5),
    alpha=0.45,
):
    """
    Plot overlaid empirical histograms for a list of estimators.

    Parameters:
    - estimators: list of callables with no argument, each returning one estimate
    - labels: list of estimator names
    - nrep: number of Monte Carlo replications
    - bins: number of histogram bins
    - figsize: matplotlib figure size
    - alpha: histogram transparency
    """
    if len(estimators) != len(labels):
        raise ValueError("estimators and labels must have the same length.")

    samples = [np.array([estimator() for _ in range(nrep)]) for estimator in estimators]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    colors = plt.cm.tab10(np.linspace(0, 1, len(estimators)))

    for values, label, color in zip(samples, labels, colors):
        ax.hist(
            values,
            bins=bins,
            alpha=alpha,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.8,
        )
        ax.axvline(np.mean(values), color=color, linestyle="--", linewidth=1.5)

    ax.set_title("Overlaid histograms of estimators")
    ax.set_xlabel("Estimate")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()

    return {
        "samples": samples,
        "means": np.array([np.mean(x) for x in samples]),
        "stds": np.array([np.std(x, ddof=1) for x in samples]),
    }







def plot_mse_proposed_vs_splitting(
    alpha,
    estimator,
    M_min=50,
    M_max=200,
    M_step=10,
    nrep=100,
    initial_L=10,
):
    """
    Compare le MSE empirique de :
    - propo_startified_estimator(alpha, M, estimator)
    - stratified_estimator_splitting(L=M, estimator, initial_L=10, n_samples=1)
    """
    if M_step <= 0:
        raise ValueError("M_step must be positive.")
    if M_max < M_min:
        raise ValueError("M_max must be greater than or equal to M_min.")

    M_grid = np.arange(M_min, M_max + 1, M_step, dtype=int)

    mse_proposed = np.zeros(len(M_grid))
    mse_splitting = np.zeros(len(M_grid))

    for i, M in enumerate(M_grid):
        proposed_samples = np.array([
            propo_startified_estimator(alpha, M, estimator)
            for _ in range(nrep)
        ])

        splitting_samples = np.array([
            stratified_estimator_splitting(M, estimator, initial_L, n_samples=1)
            for _ in range(nrep)
        ])

        mse_proposed[i] = np.var(proposed_samples)
        mse_splitting[i] = np.var(splitting_samples)

    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#4D4D4D",
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.8,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)

    sns.lineplot(
        x=M_grid,
        y=mse_proposed,
        color="#0072B2",
        marker="o",
        linewidth=2.2,
        markersize=6.0,
        ax=ax,
        label=rf"Proposed estimator ($\alpha={alpha}$)"
    )

    sns.lineplot(
        x=M_grid,
        y=mse_splitting,
        color="#D55E00",
        marker="o",
        linewidth=2.2,
        markersize=6.0,
        ax=ax,
        label=rf"Splitting estimator ($L=M$, $L_0={initial_L}$)"
    )

    ax.set_xlabel(r"$M$")
    ax.set_ylabel("Empirical MSE")
    ax.set_title("Comparison of empirical MSE")
    ax.set_xlim(np.min(M_grid), np.max(M_grid))
    ax.grid(True, axis="both")
    sns.despine(ax=ax, top=True, right=True)
    ax.legend(title=None, frameon=False, loc="best")

    plt.tight_layout()
    plt.show()

    return {
        "M_grid": M_grid,
        "mse_proposed": mse_proposed,
        "mse_splitting": mse_splitting,
    }

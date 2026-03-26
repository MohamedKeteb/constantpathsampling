import numpy as np
from src.debiasedalgo import *

def bin_variance(l0, l1, estimator, n_samples=20):
    lam = np.random.uniform(l0, l1, n_samples)
    y = np.array([estimator(l) for l in lam])
    return np.var(y)

def splitting_grid(L, estimator, initial_bins=10):
    """
    Construire une grid greedy à partir d'une grid uniform initiale.
    
    Parameters:
    - L : nombre final de bins
    - estimator : estimateur à utiliser pour calculer la variance dans chaque bin
    - initial_bins : nombre initial de bins uniformes
    """
    # Créer une grid initiale uniforme
    uniform_grid = np.linspace(0.0, 1.0, initial_bins + 1)
    bins = [(uniform_grid[i], uniform_grid[i+1]) for i in range(initial_bins)]
    
    # Greedy split jusqu'à obtenir L bins
    while len(bins) < L:
        # Calculer la variance de chaque bin
        vars_ = [bin_variance(a, b, estimator) for (a, b) in bins]
        # Choisir le bin avec variance maximale
        idx = np.argmax(vars_)
        a, b = bins[idx]
        mid = 0.5 * (a + b)
        # Remplacer le bin par ses deux moitiés
        bins.pop(idx)
        bins.insert(idx, (a, mid))
        bins.insert(idx + 1, (mid, b))
    
    # Construire la grid finale
    grid = [bins[0][0]]  # commencer par la borne inférieure du premier bin
    for (_, b) in bins:
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










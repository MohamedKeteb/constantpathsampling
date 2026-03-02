import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm  # pour la barre de progression
from src.debiasedalgo import unbiased_estimator
from src.path_MH import MH_kernel, initial_distribution, MH_coupled_kernel
# Set the random seed for reproducibility and define the parameter D

np.random.seed(42)  


def log_target0(x):
    return -0.5 * x**2
def log_target1(x, D=4):
    return -0.5 * (x - D)**2
def log_target_path(x, path):
    return (1-path) * log_target0(x) + path * log_target1(x)
def grad_log_target_path(x):
    return log_target1(x) - log_target0(x)


# --- Paramètres ---
lambda_grid = np.arange(0, 11) / 10  # 0, 0.1, ..., 1.0
nrep = 100
sigmaq = 1


def simulate_meeting_times(lambda_grid, nrep, sigmaq):

    k_grid = np.zeros(len(lambda_grid))
    meetings_list = []

    # --- Boucle sur les lambda ---
    for ilambda, lam in enumerate(tqdm(lambda_grid, desc="Lambda grid")):

        # fonctions spécifiques à lambda
        log_target = lambda x: log_target_path(x, path=lam)
        ri = lambda: initial_distribution(log_target, mean_init=-1, sigma_init=2.0)
        sk = lambda x, f: MH_kernel(x, f, sigma_proposal=sigmaq, log_target=log_target)
        ck = lambda x1, x2, f1, f2: MH_coupled_kernel(x1, x2, f1, f2, sigma_proposal=sigmaq, log_target=log_target)

        # réplications
        meetings = []
        h_list = [lambda x: x]  # h(x) = x
        for _ in range(nrep):
            uestimator = unbiased_estimator(sk, ck, ri, h_list, k=0, m=100, lag=1)
            meetings.append(uestimator["meetingtime"])

        meetings = np.array(meetings)

        # 99% quantile
        k_grid[ilambda] = np.floor(np.quantile(meetings, 0.99))

        # stocker dans un DataFrame pour plot
        df_temp = pd.DataFrame({
            "ilambda": ilambda,
            "lambda": lam,
            "meetings": meetings
        })
        meetings_list.append(df_temp)

    # --- Fusionner tous les résultats ---
    meetings_df = pd.concat(meetings_list, ignore_index=True)

    # --- Plot ---
    plt.figure(figsize=(10,6))
    sns.violinplot(x="lambda", y="meetings", data=meetings_df, inner=None, color="skyblue")
    plt.plot(lambda_grid, k_grid, color="red", marker='o', label="99% quantile")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Meeting times")
    plt.title("Meeting times for different lambdas")
    plt.legend()
    plt.show()

    return k_grid, meetings_df










import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm  # pour la barre de progression
from src.debiasedalgo import *
from src.path_MH import *


def log_target0(x):
    return -0.5 * x**2
def log_target1(x, D=4):
    return -0.5 * (x - D)**2
def log_target_path(x, path):
    return (1-path) * log_target0(x) + path * log_target1(x)
def grad_log_target_path(x):
    return log_target1(x) - log_target0(x)


# --- Paramètres ---



def tune_km_grid(lambda_grid, lag, sigmaq, nrep=100):

    k_grid = np.zeros(len(lambda_grid))
    meetings_list = []

    for ilambda, lam in enumerate(tqdm(lambda_grid, desc="Lambda grid")):

        log_target = lambda x: log_target_path(x, path=lam)

        ri = lambda: initial_distribution(
            log_target,
            mean_init=-1,
            sigma_init=2.0
        )

        sk = lambda x, f: MH_kernel(
            x, f,
            sigma_proposal=sigmaq,
            log_target=log_target
        )

        ck = lambda x1, f1, x2, f2: MH_coupled_kernel(
            x1, f1, x2, f2,
            sigma_proposal=sigmaq,
            log_target=log_target
        )

        meetings = []
        k = 0
        for _ in range(nrep):
            tau = coupled_chain(sk, ck, ri, k, lag)["meetingtime"]
            meetings.append(tau)

        meetings = np.array(meetings)

        # quantile 99%
        k_grid[ilambda] = np.floor(np.quantile(meetings, 0.99))

        df_temp = pd.DataFrame({
            "ilambda": ilambda,
            "lambda": lam,
            "meetings": meetings
        })
        meetings_list.append(df_temp)

    # --- concat ---
    meetings_df = pd.concat(meetings_list, ignore_index=True)

    # --- moyenne des meeting times ---
    meantau = (
        meetings_df
        .groupby(['ilambda', 'lambda'])['meetings']
        .mean()
        .to_numpy()
    )

    # --- calcul de m_grid ---
    m_grid = np.floor(5 * np.max(k_grid) + np.max(meantau) - meantau)

    return {
        "k_grid": k_grid,
        "m_grid": m_grid,
        "meetings_df": meetings_df,
        "meantau": meantau
    }




def plot_km_grid(lambda_grid, lag, sigmaq, nrep):


    results = tune_km_grid(lambda_grid, lag, sigmaq, nrep)
    k_grid = results["k_grid"]
    m_grid = results["m_grid"]
    meetings_df = results["meetings_df"]
    line_df = pd.DataFrame({
        "lambda": lambda_grid,
        r"$k_\lambda$ (99% quantile)": k_grid,
        r"$m_\lambda$ (constant cost)": m_grid
    }).melt(id_vars="lambda", var_name="Parameter", value_name="Value")

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

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), dpi=150)

    sns.violinplot(
        data=meetings_df,
        x="lambda",
        y="meetings",
        inner=None,
        cut=0,
        color="#DCEAF4",
        linewidth=0.8,
        ax=axes[0]
    )
    sns.lineplot(
        x=np.arange(len(lambda_grid)),
        y=k_grid,
        color="#0072B2",
        marker="o",
        linewidth=2.2,
        markersize=6.5,
        ax=axes[0],
        label=r"$k_\lambda$"
    )

    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel("Meeting time")
    axes[0].set_title("Meeting-time distribution and tuned $k$ for each $\lambda^{[\ell]}$")
    axes[0].grid(True, axis="y")
    axes[0].grid(False, axis="x")
    sns.despine(ax=axes[0], top=True, right=True)
    axes[0].legend(frameon=False, loc="upper left")

    sns.lineplot(
        data=line_df,
        x="lambda",
        y="Value",
        hue="Parameter",
        style="Parameter",
        markers=True,
        dashes=False,
        linewidth=2.2,
        markersize=6.5,
        palette=["#0072B2", "#D55E00"],
        ax=axes[1]
    )

    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Tuned k and m for each $\lambda^{[\ell]}$")
    axes[1].set_xlim(np.min(lambda_grid), np.max(lambda_grid))
    axes[1].grid(True, axis="both")
    sns.despine(ax=axes[1], top=True, right=True)
    axes[1].legend(title=None, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.92))

    plt.tight_layout()
    plt.show()





def estimate_mom(lambda_grid, k_grid, m_grid, nrep, sigmaq, lag):

    m1 = np.zeros(len(lambda_grid))
    m2 = np.zeros(len(lambda_grid))
    variance = np.zeros(len(lambda_grid))
    variance_ci_low = np.zeros(len(lambda_grid))
    variance_ci_high = np.zeros(len(lambda_grid))
    averagecost = np.zeros(len(lambda_grid))
    h_list = [lambda x : grad_log_target_path(x), lambda x: grad_log_target_path(x)**2]


    # --- Boucle sur les lambda ---
    for ilambda, lam in enumerate(tqdm(lambda_grid, desc="Final estimation")):

        # même structure que dans tune_km
        log_target = lambda x: log_target_path(x, path=lam)

        ri = lambda: initial_distribution(
            log_target,
            mean_init=-1,
            sigma_init=2.0
        )

        sk = lambda x, f: MH_kernel(
            x, f,
            sigma_proposal=sigmaq,
            log_target=log_target
        )

        ck = lambda x1, f1, x2, f2: MH_coupled_kernel(
            x1, f1, x2, f2,
            sigma_proposal=sigmaq,
            log_target=log_target
        )

        # --- Réplications ---
        uestimators = []
        for _ in range(nrep):
            res = unbiased_estimator(
                sk, ck, ri,
                h_list,
                k=int(k_grid[ilambda]),
                m=int(m_grid[ilambda]),
                lag=lag
            )
            uestimators.append(res)

        # --- Extraction ---
        mom1_estimators = np.array([
            x["uestimator"][0] for x in uestimators
        ])

        mom2_estimators = np.array([
            x["uestimator"][1] for x in uestimators
        ])

        cost_estimators = np.array([
            2 * x["meetingtime"] + max(1, x["iteration"] - x["meetingtime"] + 1)
            for x in uestimators
        ])
        variance_estimators = mom2_estimators - mom1_estimators**2

        # --- Moyennes ---
        m1[ilambda] = np.mean(mom1_estimators)
        m2[ilambda] = np.mean(mom2_estimators)
        variance[ilambda] = np.mean(variance_estimators)
        variance_ci_low[ilambda], variance_ci_high[ilambda] = np.quantile(
            variance_estimators,
            [0.025, 0.975]
        )
        averagecost[ilambda] = np.mean(cost_estimators)

    return {
        "m1": m1,
        "m2": m2,
        "variance": variance,
        "variance_ci_low": variance_ci_low,
        "variance_ci_high": variance_ci_high,
        "cost": averagecost
    }



def plot_estimate_mom(lambda_grid, k_grid, m_grid, nrep, sigmaq, lag):
    res = estimate_mom(lambda_grid, k_grid, m_grid, nrep, sigmaq, lag)
    m2 = res["m2"]
    variance = res["variance"]
    variance_ci_low = res["variance_ci_low"]
    variance_ci_high = res["variance_ci_high"]

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
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), dpi=150)

    sns.lineplot(
        x=lambda_grid,
        y=variance,
        color="#0072B2",
        marker="o",
        linewidth=2.2,
        markersize=6.5,
        ax=axes[0],
        label=r"$\widehat{\mathrm{Var}}(\hat{E}(\lambda))$"
    )
    axes[0].fill_between(
        lambda_grid,
        variance_ci_low,
        variance_ci_high,
        color="#0072B2",
        alpha=0.18,
        label=r"$95\%$ CI"
    )
    sns.lineplot(
        x=lambda_grid,
        y=np.sqrt(m2),
        color="#D55E00",
        marker="o",
        linewidth=2.2,
        markersize=6.5,
        ax=axes[1],
        label=r"$\sqrt{\hat{m}_2(\lambda)}$"
    )

    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel(r"$\widehat{\mathrm{Var}}(\hat{E}(\lambda))$")
    axes[0].set_title(r"Estimate of $\widehat{\mathrm{Var}}(\hat{E}(\lambda))$ over $\lambda^{[\ell]}$")
    axes[0].set_xlim(np.min(lambda_grid), np.max(lambda_grid))
    axes[0].grid(True, axis="both")
    sns.despine(ax=axes[0], top=True, right=True)
    axes[0].legend(title=None, frameon=False, loc="best")

    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel(r"$\sqrt{\hat{m}_2(\lambda)}$")
    axes[1].set_title(r"Estimate of $\sqrt{\hat{m}_2(\lambda)}$ over $\lambda^{[\ell]}$")
    axes[1].set_xlim(np.min(lambda_grid), np.max(lambda_grid))
    axes[1].grid(True, axis="both")
    sns.despine(ax=axes[1], top=True, right=True)
    axes[1].legend(title=None, frameon=False, loc="best")

    plt.tight_layout()
    plt.show()




def uestimator_given_lambda(lam, lambda_grid, *args):
    if len(args) != 5:
        raise TypeError(
            "uestimator_given_lambda expects "
            "(lam, lambda_grid, k_grid, m_grid, nrep, sigmaq, lag)."
        )

    k_grid, m_grid, nrep, sigmaq, lag = args
    if np.ndim(nrep) != 0 or np.ndim(sigmaq) != 0 or np.ndim(lag) != 0:
        raise TypeError(
            "Invalid arguments. Expected "
            "(lam, lambda_grid, k_grid, m_grid, nrep, sigmaq, lag). "
            "You may need to reload `src.normal` in your notebook."
        )

    # --- trouver le lambda le plus proche ---
    res = estimate_mom(lambda_grid, k_grid, m_grid, int(nrep), sigmaq, int(lag))
    m2 = res["m2"]
    m1 = res["m1"]
    index_lambda = np.argmin(np.abs(lambda_grid - lam))

    # --- paramètres associés ---
    mean_target = m1[index_lambda]
    var_target = np.maximum(m2[index_lambda] - m1[index_lambda]**2, 0.0)
    sd_target = np.sqrt(var_target)

    k = int(k_grid[index_lambda])
    m = int(m_grid[index_lambda])

    # --- log_target spécifique ---
    log_target = lambda x: log_target_path(x, path=lam)

    # --- initialisation ---
    def ri():
        chain_state = np.random.normal(loc=mean_target, scale=sd_target, size=1)
        current_pdf = log_target(chain_state)
        return chain_state, current_pdf

    # --- kernels (cohérents avec ton code MH) ---
    sk = lambda x, f: MH_kernel(
            x, f,
            sigma_proposal=sigmaq,
            log_target=log_target
        )

    ck = lambda x1, f1, x2, f2: MH_coupled_kernel(
            x1, f1, x2, f2,
            sigma_proposal=sigmaq,
            log_target=log_target
        )

    # --- appel estimateur ---
    ue_ = unbiased_estimator(
        sk, ck, ri,
        h=[lambda x: grad_log_target_path(x)],
        k=k,
        m=m,
        lag=lag
    )

    return ue_

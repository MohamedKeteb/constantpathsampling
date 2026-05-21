import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm  # pour la barre de progression
from src.debiasedalgo import *
from src.path_MH import *




# --- Paramètres ---



def tune_km_grid(lambda_grid, lag, sigmaq, log_target_path, nrep=100):

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




def plot_km_grid(lambda_grid, lag, sigmaq, log_target_path, nrep):


    results = tune_km_grid(lambda_grid, lag, sigmaq, log_target_path, nrep)
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





def estimate_mom(lambda_grid, k_grid, m_grid, nrep, sigmaq, lag, log_target_path, grad_log_target_path):

    m1 = np.zeros(len(lambda_grid))
    m2 = np.zeros(len(lambda_grid))
    m1_ci_low = np.zeros(len(lambda_grid))
    m1_ci_high = np.zeros(len(lambda_grid))
    m2_ci_low = np.zeros(len(lambda_grid))
    m2_ci_high = np.zeros(len(lambda_grid))
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
        m1_ci_low[ilambda], m1_ci_high[ilambda] = np.quantile(
            mom1_estimators,
            [0.025, 0.975]
        )
        m2_ci_low[ilambda], m2_ci_high[ilambda] = np.quantile(
            mom2_estimators,
            [0.025, 0.975]
        )
        variance[ilambda] = np.mean(variance_estimators)
        variance_ci_low[ilambda], variance_ci_high[ilambda] = np.quantile(
            variance_estimators,
            [0.025, 0.975]
        )
        averagecost[ilambda] = np.mean(cost_estimators)

    return {
        "m1": m1,
        "m2": m2,
        "m1_ci_low": m1_ci_low,
        "m1_ci_high": m1_ci_high,
        "m2_ci_low": m2_ci_low,
        "m2_ci_high": m2_ci_high,
        "variance": variance,
        "variance_ci_low": variance_ci_low,
        "variance_ci_high": variance_ci_high,
        "cost": averagecost
    }



def plot_estimate_mom(lambda_grid, k_grid, m_grid, nrep, sigmaq, lag, log_target_path, grad_log_target_path):
    res = estimate_mom(lambda_grid, k_grid, m_grid, nrep, sigmaq, lag, log_target_path, grad_log_target_path)
    m1 = res["m1"]
    m1_ci_low = res["m1_ci_low"]
    m1_ci_high = res["m1_ci_high"]
    m2 = np.sqrt(res["m2"])
    m2_ci_low = np.sqrt(np.maximum(res["m2_ci_low"], 0.0))
    m2_ci_high = np.sqrt(np.maximum(res["m2_ci_high"], 0.0))
    sqrt_variance = np.sqrt(np.maximum(res["variance"], 0.0))
    sqrt_variance_ci_low = np.sqrt(np.maximum(res["variance_ci_low"], 0.0))
    sqrt_variance_ci_high = np.sqrt(np.maximum(res["variance_ci_high"], 0.0))

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
        y=sqrt_variance,
        color="#0072B2",
        marker="o",
        linewidth=2.2,
        markersize=6.5,
        ax=axes[0],
        label=r"$\sqrt{\widehat{\mathrm{Var}}(\hat{E}(\lambda))}$"
    )
    axes[0].fill_between(
        lambda_grid,
        sqrt_variance_ci_low,
        sqrt_variance_ci_high,
        color="#0072B2",
        alpha=0.18,
        label=r"$95\%$ CI for $\sqrt{\widehat{\mathrm{Var}}(\hat{E}(\lambda))}$"
    )
    sns.lineplot(
        x=lambda_grid,
        y=m2,
        color="#D55E00",
        marker="o",
        linewidth=2.2,
        markersize=6.5,
        ax=axes[0],
        label=r"$\sqrt{\hat{m}_2(\lambda)}$"
    )
    axes[0].fill_between(
        lambda_grid,
        m2_ci_low,
        m2_ci_high,
        color="#D55E00",
        alpha=0.18,
        label=r"$95\%$ CI for $\sqrt{\hat{m}_2(\lambda)}$"
    )

    sns.lineplot(
        x=lambda_grid,
        y=m1,
        color="#009E73",
        marker="o",
        linewidth=2.2,
        markersize=6.5,
        ax=axes[1],
        label=r"$\hat{m}_1(\lambda)$"
    )
    axes[1].fill_between(
        lambda_grid,
        m1_ci_low,
        m1_ci_high,
        color="#009E73",
        alpha=0.18,
        label=r"$95\%$ CI"
    )

    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel(r"Square-root scale")
    axes[0].set_title(
        r"Estimates of $\sqrt{\widehat{\mathrm{Var}}(\hat{E}(\lambda))}$ and "
        r"$\sqrt{\hat{m}_2(\lambda)}$ over $\lambda^{[\ell]}$"
    )
    axes[0].set_xlim(np.min(lambda_grid), np.max(lambda_grid))
    axes[0].grid(True, axis="both")
    sns.despine(ax=axes[0], top=True, right=True)
    axes[0].legend(title=None, frameon=False, loc="best")

    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel(r"$\hat{m}_1(\lambda)$")
    axes[1].set_title(r"Estimate of $\hat{m}_1(\lambda)$ over $\lambda^{[\ell]}$")
    axes[1].set_xlim(np.min(lambda_grid), np.max(lambda_grid))
    axes[1].grid(True, axis="both")
    sns.despine(ax=axes[1], top=True, right=True)
    axes[1].legend(title=None, frameon=False, loc="best")

    plt.tight_layout()
    plt.show()




def uestimator_given_lambda(lam, lambda_grid, k_grid, m_grid, sigmaq, lag, log_target_path, grad_log_target_path):

    # --- trouver le lambda le plus proche ---
    #res = estimate_mom(lambda_grid, k_grid, m_grid, int(nrep), sigmaq, int(lag))
    #m2 = res["m2"]
    #m1 = res["m1"]
    index_lambda = np.argmin(np.abs(lambda_grid - lam))

    # --- paramètres associés ---
    #mean_target m1[index_lambda]
    #var_target = np.maximum(m2[index_lambda] - m1[index_lambda]**2, 0.0)
    #sd_target = np.sqrt(var_target)

    k = int(k_grid[index_lambda])
    m = int(m_grid[index_lambda])

    # --- log_target spécifique ---
    log_target = lambda x: log_target_path(x, path=lam)

    # --- initialisation ---
    def ri():
        chain_state = np.random.normal(loc=-1, scale=2, size=1)
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

    return ue_["uestimator"][0]


def q_unbiased_estimator(L, M, estimator, sqrt_m2):

    grid = np.linspace(0.0, 1.0, L + 1)
    widths = np.diff(grid)

    # Hauteur de la marche sur chaque intervalle :
    # proportionnelle à (sqrt_m2(l) + sqrt_m2(l+1)) / 2
    step_heights = np.array([
        0.5 * (sqrt_m2[l] + sqrt_m2[l + 1])
        for l in range(L)
    ], dtype=float)

    # Normalisation pour obtenir une densité
    total_mass = np.sum(step_heights * widths)
    normalized_heights = step_heights / total_mass
    interval_masses = normalized_heights * widths

    def sqrt_m2_estimate(x):
        idx = min(int(np.floor(x * L)), L - 1)
        return normalized_heights[idx]

    def sample_sqrt_m2():
        # Choix de l'intervalle selon les masses, puis uniforme dans cet intervalle
        idx = np.random.choice(L, p=interval_masses)
        a = grid[idx]
        b = grid[idx + 1]
        x = np.random.uniform(a, b)
        return x

    res = 0.0
    for _ in range(M):
        lam = sample_sqrt_m2()
        q_estimate = sqrt_m2_estimate(lam)
        res += estimator(lam) / q_estimate

    return res / M


import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotnine as p9
from tqdm.auto import tqdm

from unbiased_mcmc.IS import IS_estimator
from unbiased_mcmc.SUPS import SUPS_estimator, neyman_allocation
from unbiased_mcmc.UPS import E_lambda
from unbiased_mcmc.UPS import UPS_estimator


def compare_estimators(
	lambda_grid,
	m2,
	sigma,
	k_grid,
	m_grid,
	lag,
	mean_init,
	Sigma_init,
	Sigma,
	cholesky_Sigma,
	inv_cholesky_Sigma,
	log_target_path,
	grad_log_target_path,
	log_target0,
	log_target1,
	D,
	d,
	M=100,
	nrep=500,
	true_value=0.0,
	n_workers=14,
):
	

	max_workers = min(os.cpu_count() or 1, n_workers, 2 * nrep)
	budget = neyman_allocation(lambda_grid, sigma, M)

	def run_ups(_):
		estimate, cost = UPS_estimator(
			lambda_grid,
			m2,
			k_grid,
			m_grid,
			lag,
			mean_init,
			Sigma_init,
			Sigma,
			cholesky_Sigma,
			inv_cholesky_Sigma,
			log_target_path,
			grad_log_target_path,
			M,
		)
		return float(np.asarray(estimate)), float(np.asarray(cost))

	def run_sups(_):
		estimate, cost = SUPS_estimator(
			lambda_grid,
			budget,
			k_grid,
			m_grid,
			lag,
			mean_init,
			Sigma_init,
			Sigma,
			cholesky_Sigma,
			inv_cholesky_Sigma,
			log_target_path,
			grad_log_target_path,
		)
		return float(np.asarray(estimate)), float(np.asarray(cost))

	ups_results = [None] * nrep
	sups_results = [None] * nrep
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		future_methods = {}
		for index in range(nrep):
			future_methods[executor.submit(run_ups, index)] = ("UPS", index)
			future_methods[executor.submit(run_sups, index)] = ("SUPS", index)

		with tqdm(total=nrep, desc="UPS", position=0) as ups_progress, tqdm(
			total=nrep, desc="SUPS", position=1
		) as sups_progress:
			for future in as_completed(future_methods):
				method, index = future_methods[future]
				result = future.result()
				if method == "UPS":
					ups_results[index] = result
					ups_progress.update(1)
				else:
					sups_results[index] = result
					sups_progress.update(1)

	ups_values = np.asarray([result[0] for result in ups_results])
	ups_costs = np.asarray([result[1] for result in ups_results])
	sups_values = np.asarray([result[0] for result in sups_results])
	sups_costs = np.asarray([result[1] for result in sups_results])

	M_is = max(1, int(round((ups_costs.mean() + sups_costs.mean()) / 2)))
	log_weight = lambda x: log_target1(x, D) - log_target0(x)
	proposal = lambda: np.random.normal(size=d)
	is_values = [None] * nrep
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		is_futures = {
			executor.submit(IS_estimator, log_weight, proposal, M_is): index
			for index in range(nrep)
		}
		for future in tqdm(
			as_completed(is_futures), total=nrep, desc="IS replications"
		):
			is_values[is_futures[future]] = future.result()
	is_values = np.asarray(is_values)

	comparison_mse = {
		"UPS": np.mean((ups_values - true_value) ** 2),
		"SUPS": np.mean((sups_values - true_value) ** 2),
		"IS": np.mean((is_values - true_value) ** 2),
	}
	results = {
		"UPS_values": ups_values,
		"UPS_costs": ups_costs,
		"SUPS_values": sups_values,
		"SUPS_costs": sups_costs,
		"IS_values": is_values,
		"M_is": M_is,
		"MSE": comparison_mse,
		"budget": budget,
		"n_workers": max_workers,
	}

	plot_data = pd.DataFrame(
		{
			"Estimate": np.concatenate([ups_values, sups_values, is_values]),
			"Estimator": np.repeat(["UPS", "SUPS", "IS"], nrep),
		}
	)
	plot_data["Estimator"] = pd.Categorical(
		plot_data["Estimator"],
		categories=["UPS", "SUPS", "IS"],
		ordered=True,
	)
	true_value_label = pd.DataFrame(
		{
			"Estimator": ["IS"],
			"Estimate": [true_value],
			"label": ["True value"],
		}
	)
	comparison_plot = (
		p9.ggplot(plot_data, p9.aes(x="Estimator", y="Estimate"))
		+ p9.geom_boxplot(width=0.6, fill="white", color="black", size=0.5)
		+ p9.geom_hline(
			yintercept=true_value,
			linetype="dashed",
			color="#b44b35",
			size=0.5,
		)
		+ p9.geom_text(
			data=true_value_label,
			mapping=p9.aes(x="Estimator", y="Estimate", label="label"),
			inherit_aes=False,
			ha="center",
			va="bottom",
			color="#b44b35",
			size=12,
		)
		+ p9.labs(x="", y="Estimate", title=None)
		+ p9.theme_classic(base_size=11)
		+ p9.theme(figure_size=(6, 4))
	)
	results["plot"] = comparison_plot

	print(
		f"Workers={max_workers} | M_IS={M_is} | "
		f"coût moyen UPS={ups_costs.mean():.1f} | "
		f"coût moyen SUPS={sups_costs.mean():.1f}"
	)
	print("MSE:", comparison_mse)
	return results


def plot_E_lambda_over_q(
	lambda_grid,
	m2,
	k_grid,
	m_grid,
	lag,
	mean_init,
	Sigma_init,
	Sigma,
	cholesky_Sigma,
	inv_cholesky_Sigma,
	log_target_path,
	grad_log_target_path,
	nrep=10,
):
	lambda_grid = np.asarray(lambda_grid, dtype=float)
	m2 = np.asarray(m2, dtype=float)

	widths = np.diff(lambda_grid)
	heights = 0.5 * (
		 np.sqrt(np.maximum(m2[:-1], 0.0))
		+ np.sqrt(np.maximum(m2[1:], 0.0))
	)
	weights = widths * heights
	normalizer = np.sum(weights)
	if not np.isfinite(normalizer) or normalizer <= 0:
		raise ValueError("m2 must define a finite, nonzero proposal density")
	interval_densities = heights / normalizer
	interval_indices = np.clip(
		np.searchsorted(lambda_grid, lambda_grid, side="right") - 1,
		0,
		widths.size - 1,
	)
	q_values = interval_densities[interval_indices]

	mean_estimates = []
	for lam in lambda_grid:
		replicates = np.asarray(
			[
				E_lambda(
					lam,
					lambda_grid,
					k_grid,
				m_grid,
				lag,
				mean_init,
				Sigma_init,
				Sigma,
				cholesky_Sigma,
				inv_cholesky_Sigma,
				log_target_path,
				grad_log_target_path,
				)[0]
				for _ in range(nrep)
			]
		)
		mean_estimates.append(np.mean(replicates, axis=0))

	estimates = np.asarray(mean_estimates)
	if estimates.ndim == 1:
		values = estimates / q_values
		data = pd.DataFrame(
			{"lambda": lambda_grid, "estimate": values}
		)
		plot = p9.ggplot(data, p9.aes(x="lambda", y="estimate"))
	else:
		components = estimates.reshape(estimates.shape[0], -1)
		values = components / q_values[:, None]
		data = pd.DataFrame(
			{
				"lambda": np.tile(lambda_grid, components.shape[1]),
				"estimate": values.T.reshape(-1),
				"component": np.repeat(
					[np.arange(components.shape[1]) + 1], lambda_grid.size, axis=0
				).T.reshape(-1),
			}
		)
		plot = p9.ggplot(
			data,
			p9.aes(x="lambda", y="estimate", group="component"),
		)
		plot += p9.facet_wrap("~component", ncol=1, scales="free_y")

	return (
		plot
		+ p9.geom_line(color="black", size=0.6)
		+ p9.geom_point(color="black", fill="white", size=1.5)
		+ p9.labs(x=r"$\lambda$", y=r"$\widehat{E}(\lambda) / \nu(\lambda)$", title=None)
		+ p9.theme_classic(base_size=11)
		+ p9.theme(
			figure_size=(6, 4),
			panel_background=p9.element_rect(fill="white", color="white"),
			plot_background=p9.element_rect(fill="white", color="white"),
		)
	)

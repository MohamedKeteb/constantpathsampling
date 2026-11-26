import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotnine as p9
from tqdm.auto import tqdm

from unbiased_mcmc.IS import IS_estimator
from unbiased_mcmc.SUPS import SUPS_estimator, neyman_allocation, estimate_sigma
from unbiased_mcmc.UPS import E_lambda
from unbiased_mcmc.UPS import UPS_estimator
from unbiased_mcmc.tune_umcmc import tune_km_grid, estimate_moments


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
	batch_log_weight = lambda samples: log_target1(samples, D) - log_target0(samples)
	batch_proposal = lambda sample_count: np.random.normal(size=(sample_count, d))
	is_values = [None] * nrep
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		is_futures = {
			executor.submit(
				IS_estimator,
				log_weight,
				proposal,
				M_is,
				batch_log_weight=batch_log_weight,
				batch_proposal=batch_proposal,
			): index
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
			size=18,
		)
		+ p9.labs(x="", y="", title=None)
		+ p9.theme_classic(base_size=15)
		+ p9.theme(
			figure_size=(6, 4),
			axis_text_x=p9.element_text(size=18),
		)
	)
	results["plot"] = comparison_plot

	print(
		f"Workers={max_workers} | M_IS={M_is} | "
		f"coût moyen UPS={ups_costs.mean():.1f} | "
		f"coût moyen SUPS={sups_costs.mean():.1f}"
	)
	print("MSE:", comparison_mse)
	return results


def compare_estimators_mse_by_budget(
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
	M_values,
	nrep,
	true_value=0.0,
	n_workers=14,
	make_plots=True,
):
	

	max_workers = min(os.cpu_count() or 1, n_workers, 2 * nrep)
	records = []
	replicates = {}
	budget_results = {}

	for M in M_values:
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

		method_runs = {"UPS": run_ups, "SUPS": run_sups}
		method_results = {method: [None] * nrep for method in method_runs}
		with ThreadPoolExecutor(max_workers=max_workers) as executor:
			future_methods = {
				executor.submit(run, index): (method, index)
				for method, run in method_runs.items()
				for index in range(nrep)
			}
			for future in tqdm(
				as_completed(future_methods),
				total=len(future_methods),
				desc=f"M={M} UPS/SUPS",
			):
				method, index = future_methods[future]
				method_results[method][index] = future.result()

		ups_values = np.asarray([result[0] for result in method_results["UPS"]])
		ups_costs = np.asarray([result[1] for result in method_results["UPS"]])
		sups_values = np.asarray([result[0] for result in method_results["SUPS"]])
		sups_costs = np.asarray([result[1] for result in method_results["SUPS"]])
		M_is = max(1, int(round((ups_costs.mean() + sups_costs.mean()) / 2)))
		budget_results[int(M)] = {
			"UPS": (ups_values, float(ups_costs.mean())),
			"SUPS": (sups_values, float(sups_costs.mean())),
			"M_is": M_is,
		}

	M_is_by_budget = {
		budget: result["M_is"] for budget, result in budget_results.items()
	}
	max_M_is = max(M_is_by_budget.values())
	batch_log_weight = lambda samples: log_target1(samples, D) - log_target0(samples)
	batch_proposal = lambda sample_count: np.random.normal(size=(sample_count, d))

	def run_is_replication(_):
		samples = batch_proposal(max_M_is)
		log_weights = np.asarray(batch_log_weight(samples), dtype=float)
		if log_weights.shape != (max_M_is,):
			raise ValueError("batch log weight must return one value per IS sample")
		cumulative_log_sums = np.logaddexp.accumulate(log_weights)
		return {
			budget: float(cumulative_log_sums[M_is - 1] - np.log(M_is))
			for budget, M_is in M_is_by_budget.items()
		}

	is_replications = [None] * nrep
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		is_futures = {
			executor.submit(run_is_replication, index): index
			for index in range(nrep)
		}
		for future in tqdm(
			as_completed(is_futures),
			total=nrep,
			desc=f"IS prefixes (max M={max_M_is})",
		):
			is_replications[is_futures[future]] = future.result()

	for M in M_values:
		budget = int(M)
		M_is = M_is_by_budget[budget]
		is_values = np.asarray(
			[replication[budget] for replication in is_replications], dtype=float
		)
		method_values = {
			method: result
			for method, result in budget_results[budget].items()
			if method != "M_is"
		}
		method_values["IS"] = (is_values, float(M_is))
		replicates[budget] = {
			method: values.copy() for method, (values, _) in method_values.items()
		}
		for method, (values, mean_cost) in method_values.items():
			bias_squared = float((np.mean(values) - true_value) ** 2)
			variance = float(np.var(values))
			records.append(
				{
					"M": budget,
					"Estimator": method,
					"Cost": float(mean_cost),
					"Bias squared": bias_squared,
					"Variance": variance,
					"MSE": bias_squared + variance,
					"M_is": M_is,
				}
			)

	results_data = pd.DataFrame(records)
	plots = {}
	for metric in (("Bias squared", "Variance", "MSE") if make_plots else ()):
		metric_data = results_data.loc[
			(results_data["Cost"] > 0) & (results_data[metric] > 0)
		].copy()
		min_exponent = int(np.floor(np.log10(metric_data["Cost"].min())))
		max_exponent = int(np.ceil(np.log10(metric_data["Cost"].max())))
		if min_exponent == max_exponent:
			max_exponent += 1
		cost_breaks = np.power(10.0, np.arange(min_exponent, max_exponent + 1))
		shape_labels = {"UPS": "UPS", "SUPS": "SUPS", "IS": "IS"}
		if metric == "MSE":
			for estimator, estimator_data in metric_data.groupby("Estimator"):
				if len(estimator_data) >= 2:
					slope = np.polyfit(
						np.log10(estimator_data["Cost"]),
						np.log10(estimator_data[metric]),
						1,
					)[0]
					shape_labels[estimator] = f"{estimator} (slope={slope:.2f})"
		plot = (
			p9.ggplot(
				metric_data,
				p9.aes(x="Cost", y=metric, group="Estimator"),
			)
			+ p9.geom_line(color="black")
			+ p9.geom_point(p9.aes(shape="Estimator"), color="black", size=2)
			+ p9.scale_shape_manual(
				values={"UPS": "s", "SUPS": "o", "IS": "^"},
				labels=shape_labels,
			)
			+ p9.scale_x_log10(
				breaks=cost_breaks,
				limits=(cost_breaks[0], cost_breaks[-1]),
			)
			+ p9.scale_y_log10()
			+ p9.labs(
				x="Average cost",
				y="",
				shape="",
			)
			+ p9.theme_classic(base_size=11)
			+ p9.theme(
				figure_size=(6, 4),
				axis_text_x=p9.element_text(size=14),
				axis_title_x=p9.element_text(size=15),
			)
		)
		label_data = metric_data.loc[
			(metric_data["Estimator"] == "IS")
			& metric_data["M"].isin([20, M_values[-1]])
		].copy()
		label_data["M_label"] = label_data["M"].map(
			lambda value: "M=20" if value == 20 else str(value)
		)
		label_data["label_y"] = label_data[metric] * np.where(
			np.arange(len(label_data)) % 2 == 0, 1.75, 1 / 1.75
		)
		plot += p9.geom_text(
			data=label_data,
			mapping=p9.aes(y="label_y", label="M_label"),
			va="center",
			size=15,
			inherit_aes=True,
			show_legend=False,
		)
		#if metric == "Bias squared":
		plot += p9.theme(
				legend_position=(0.02, 0.02) if metric == "MSE" else (0.02, 0.6),
				legend_justification=(0, 0) if metric == "MSE" else (0, 1),
				legend_text=p9.element_text(size=15),
			)
		#else:
		#	plot += p9.theme(legend_position="none")
		plots[metric] = plot
	return {"data": results_data, "replicates": replicates, "plots": plots}


def plot_mse_by_alpha(
	lambda_grid,
	Sigma,
	mean_init,
	Sigma_init,
	cholesky_Sigma,
	inv_cholesky_Sigma,
	log_target0,
	log_target1,
	alpha_values=(0.5, 1, 2, 3),
	M=100,
	nrep=50,
	n_tune=100,
	n_moments=100,
	n_pilot=100,
	n_workers=14,
):
	"""Compare UPS, SUPS and IS MSE across mean shifts in dimension two."""
	d = 2
	rows = []
	alpha_results = {}

	for alpha in alpha_values:
		D = alpha * np.ones(d)
		log_target_path = lambda x, path, D=D: (
			(1 - path) * log_target0(x) + path * log_target1(x, D)
		)
		grad_log_target_path = lambda x, path, D=D: (
			log_target1(x, D) - log_target0(x)
		)
		km = tune_km_grid(
			lambda_grid,
			mean_init,
			Sigma_init,
			Sigma,
			cholesky_Sigma,
			inv_cholesky_Sigma,
			log_target_path,
			lag=1,
			nrep=n_tune,
		)
		k_grid = km["k_grid"]
		m_grid = km["m_grid"]
		lag = k_grid
		moment_results = estimate_moments(
			lambda_grid,
			k_grid,
			m_grid,
			mean_init,
			Sigma_init,
			Sigma,
			cholesky_Sigma,
			inv_cholesky_Sigma,
			log_target_path,
			grad_log_target_path,
			lag=lag,
			nrep=n_moments,
		)
		sigma = estimate_sigma(
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
			N_pilot=n_pilot,
		)
		comparison = compare_estimators_mse_by_budget(
			lambda_grid,
			moment_results["m2"],
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
			M_values=(M,),
			nrep=nrep,
			true_value=0.0,
			n_workers=n_workers,
			make_plots=False,
		)
		data = comparison["data"].copy()
		data["alpha"] = alpha
		rows.append(data)
		alpha_results[alpha] = comparison

	results_data = pd.concat(rows, ignore_index=True)
	plot = (
		p9.ggplot(
			results_data,
			p9.aes(x="alpha", y="MSE", group="Estimator"),
		)
		+ p9.geom_line(color="black")
		+ p9.geom_point(p9.aes(shape="Estimator"), color="black", size=2)
		+ p9.scale_shape_manual(
			values={"UPS": "s", "SUPS": "o", "IS": "^"},
			labels={"UPS": "UPS", "SUPS": "SUPS", "IS": "IS"},
		)
		+ p9.scale_x_continuous(breaks=list(alpha_values))
		+ p9.scale_y_log10()
		+ p9.labs(x=r"$\alpha$", y="MSE", shape="")
		+ p9.theme_classic(base_size=11)
		+ p9.theme(
			figure_size=(6, 4),
			legend_position=(0.98, 0.02),
			legend_justification=(1, 0),
			legend_text=p9.element_text(size=15),
			axis_text_x=p9.element_text(size=14),
			axis_title_x=p9.element_text(size=15),
		)
	)
	return {"data": results_data, "plot": plot, "by_alpha": alpha_results}


def compare_estimators_mse_by_dimension(
	lambda_grid,
	log_target0,
	log_target1,
	dimensions=(2, 4, 8, 16, 32),
	M_by_dimension= {2: 50, 4: 100, 8: 200, 16: 500, 32: 1000},
	alpha=3.0,
	nrep=50,
	lag=1,
	init_mean=-2.0,
	init_variance=1.0,
	n_tune=100,
	n_pilot=100,
	n_moments=100,
	n_workers=14,
):
	"""Compare SUPS and IS error metrics against dimension for Gaussian shifts.

	The mean shift is ``alpha * ones(d)``. Each dimension uses its corresponding
	budget from ``M_by_dimension``. The IS sample count is set by the average
	UPS/SUPS cost over ``nrep`` replications at that dimension and budget.
	"""

	rows = []
	dimension_results = {}
	for dimension in dimensions:
		d = int(dimension)
		if d not in M_by_dimension:
			raise ValueError(f"Missing M budget for dimension d={d}")
		M = int(M_by_dimension[d])
		D = alpha * np.ones(d)
		Sigma = np.eye(d)
		Sigma_init = init_variance * np.eye(d)
		mean_init_d = init_mean * np.ones(d)
		cholesky_Sigma = np.linalg.cholesky(Sigma)
		inv_cholesky_Sigma = np.linalg.inv(cholesky_Sigma)
		log_target_path = lambda x, path, D=D: (
			(1 - path) * log_target0(x) + path * log_target1(x, D)
		)
		grad_log_target_path = lambda x, path, D=D: (
			log_target1(x, D) - log_target0(x)
		)
		km = tune_km_grid(
			lambda_grid,
			mean_init_d,
			Sigma_init,
			Sigma,
			cholesky_Sigma,
			inv_cholesky_Sigma,
			log_target_path,
			lag=lag,
			nrep=n_tune,
		)
		k_grid = km["k_grid"]
		m_grid = km["m_grid"]
		moments = estimate_moments(
			lambda_grid,
			k_grid,
			m_grid,
			mean_init_d,
			Sigma_init,
			Sigma,
			cholesky_Sigma,
			inv_cholesky_Sigma,
			log_target_path,
			grad_log_target_path,
			lag=lag,
			nrep=n_moments,
		)
		sigma = estimate_sigma(
			lambda_grid,
			k_grid,
			m_grid,
			lag,
			mean_init_d,
			Sigma_init,
			Sigma,
			cholesky_Sigma,
			inv_cholesky_Sigma,
			log_target_path,
			grad_log_target_path,
			N_pilot=n_pilot,
		)
		comparison = compare_estimators_mse_by_budget(
			lambda_grid,
			moments["m2"],
			sigma,
			k_grid,
			m_grid,
			lag,
			mean_init_d,
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
			M_values=(M,),
			nrep=nrep,
			true_value=0.0,
			n_workers=n_workers,
			make_plots=False,
		)
		data = comparison["data"].copy()
		data["Dimension"] = d
		rows.append(data)
		dimension_results[d] = comparison

	results_data = pd.concat(rows, ignore_index=True)
	results_data["Estimator"] = pd.Categorical(
		results_data["Estimator"],
		categories=["UPS", "SUPS", "IS"],
		ordered=True,
	)
	plots = {}
	for metric in ("Bias squared", "Variance", "MSE"):
		metric_data = results_data.loc[results_data[metric] > 0].copy()
		plot = (
			p9.ggplot(
				metric_data,
				p9.aes(
					x="Dimension",
					y=metric,
					group="Estimator",
					shape="Estimator",
				),
			)
			+ p9.geom_line(color="black")
			+ p9.geom_point(color="black", size=2)
			+ p9.scale_shape_manual(
				values={"UPS": "s", "SUPS": "o", "IS": "^"},
				limits=["UPS", "SUPS", "IS"],
			)
			+ p9.scale_x_continuous(breaks=list(dimensions))
			+ p9.scale_y_log10()
			+ p9.labs(x="Dimension", y=metric, shape="")
			+ p9.theme_classic(base_size=11)
			+ p9.theme(
				figure_size=(6, 4),
				axis_text_x=p9.element_text(size=14),
				axis_title_x=p9.element_text(size=15),
				legend_text=p9.element_text(size=15),
				legend_position=(0.98, 0.02),
				legend_justification=(1, 0),
			)
		)
		plots[metric] = plot
	return {
		"data": results_data,
		"plots": plots,
		"by_dimension": dimension_results,
	}


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
		+ p9.labs(x=r"$\lambda$", y="", title=None)
		+ p9.theme_classic(base_size=15)
		+ p9.theme(
			figure_size=(6, 4),
			axis_text_x=p9.element_text(size=15),
			axis_title_x=p9.element_text(size=15),
			panel_background=p9.element_rect(fill="white", color="white"),
			plot_background=p9.element_rect(fill="white", color="white"),
		)
	)

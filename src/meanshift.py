import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from matplotlib.patches import Patch

from src.IS_estimator import IS_estimator
from src.SUPS_estimator import SUPS_estimator, estimate_sigma, neyman_allocation
from src.UPS_estimator import UPS_estimator, E_lambda, optimal_q
from src.tuning_umcmc import estimate_moments, tune_km_grid


@jax.jit
def _log_target0(x):
	return -0.5 * jnp.sum(x**2)


@jax.jit
def _log_target1(x, D):
	return -0.5 * jnp.sum((x - D) ** 2)


@jax.jit
def _g_log_target(x, path, D):
	return (1 - path) * _log_target0(x) + path * _log_target1(x, D)


@jax.jit
def _g_grad_log_target(x, D):
	return _log_target1(x, D) - _log_target0(x)


def plot_ups_integrand_and_proposal(
	key,
	dimension,
	lambda_grid,
	alpha=10,
	n_rep_gridpoint=10,
	n_rep_tuning=20,
	n_rep_moments=20,
	n_proposal_samples=10000,
):
	"""Plot the averaged E(lambda) / q(lambda) and UPS proposal density."""

	n_proposal_samples = int(n_proposal_samples)
	lambda_grid = jnp.asarray(lambda_grid)
	

	key_tuning, key_moments, key_estimates = jax.random.split(key, 3)
	Sigma = jnp.eye(dimension)
	mean_init = -jnp.ones(dimension)
	Sigma_init = jnp.eye(dimension)
	D = alpha * jnp.ones(dimension)
	log_target_path = lambda x, path: _g_log_target(x, path, D)
	grad_log_target_path = lambda x, path: _g_grad_log_target(x, D)

	tuning = tune_km_grid(
		key_tuning,
		lambda_grid,
		Sigma,
		mean_init,
		Sigma_init,
		log_target_path,
		lag=1,
		nrep=n_rep_tuning,
	)
	k_grid = tuning["k_grid"]
	m_grid = tuning["m_grid"]
	lag = k_grid
	moment_results = estimate_moments(
		key_moments,
		lambda_grid,
		k_grid,
		m_grid,
		Sigma,
		mean_init,
		Sigma_init,
		log_target_path,
		grad_log_target_path,
		lag=1,
		nrep=n_rep_moments,
	)
	m2 = moment_results["m2"]

	widths = jnp.diff(lambda_grid)
	sqrt_m2 = jnp.sqrt(jnp.maximum(m2, 0.0))
	interval_weights = widths * 0.5 * (sqrt_m2[:-1] + sqrt_m2[1:])
	weight_sum = jnp.sum(interval_weights)
	if not bool(jax.device_get(jnp.isfinite(weight_sum) & (weight_sum > 0))):
		raise ValueError("m2 must define a finite, nonzero proposal density.")
	interval_weights = interval_weights / weight_sum
	proposal_density = interval_weights / widths

	replication_keys = jax.random.split(
		key_estimates, lambda_grid.shape[0] * n_rep_gridpoint
	)
	integrand_estimates = []
	for grid_index, lam in enumerate(lambda_grid):
		start = grid_index * n_rep_gridpoint
		grid_keys = replication_keys[start : start + n_rep_gridpoint]
		grid_estimates = [
			E_lambda(
				rep_key,
				lam,
				lambda_grid,
				k_grid,
				m_grid,
				lag,
				mean_init,
				Sigma_init,
				Sigma,
				log_target_path,
				grad_log_target_path,
			)
			for rep_key in grid_keys
		]
		mean_e_lambda = jnp.mean(jnp.stack(grid_estimates))
		# At the final grid point, use the density of the final interval.
		interval_index = min(grid_index, lambda_grid.shape[0] - 2)
		integrand_estimates.append(
			mean_e_lambda / proposal_density[interval_index]
		)

	integrand_estimates = jnp.stack(integrand_estimates)
	proposal_keys = jax.random.split(key_estimates, n_proposal_samples)
	proposal_samples, _ = jax.vmap(
		lambda proposal_key: optimal_q(proposal_key, lambda_grid, m2)
	)(proposal_keys)
	lambda_values = np.asarray(jax.device_get(lambda_grid))
	fig, ax = plt.subplots(figsize=(7, 4.5))
	proposal_ax = ax.twinx()
	ax.plot(
		lambda_values,
		jax.device_get(integrand_estimates),
		marker="o",
		linewidth=2,
		color="#0072B2",
		label=r"$E(\lambda)/\nu(\lambda)$",
	)
	proposal_ax.hist(
		jax.device_get(proposal_samples),
		bins=lambda_values,
		density=True,
		color="#D55E00",
		alpha=0.45,
		edgecolor="white",
		label=r"$\nu(\lambda)$",
	)
	ax.set(
		xlabel=r"$\lambda$",
		ylabel=r"$E(\lambda)/\nu(\lambda)$",
	)
	proposal_ax.set_ylabel(r"Proposal density $\nu(\lambda)$", color="#D55E00")
	proposal_ax.tick_params(axis="y", labelcolor="#D55E00")
	handles, labels = ax.get_legend_handles_labels()
	proposal_handles, proposal_labels = proposal_ax.get_legend_handles_labels()
	ax.legend(handles + proposal_handles, labels + proposal_labels)
	sns.despine(ax=ax)
	fig.tight_layout()
	plt.show()

	return {
		"lambda_grid": lambda_values,
		"integrand_estimates": np.asarray(jax.device_get(integrand_estimates)),
		"proposal_density": np.asarray(jax.device_get(proposal_density)),
		"proposal_samples": np.asarray(jax.device_get(proposal_samples)),
		"m2": np.asarray(jax.device_get(m2)),
		"figure": fig,
		"axes": ax,
		"proposal_axes": proposal_ax,
	}


def inefficiency_plot(
	key,
	dimensions,
	M_values,
	lambda_grid,
	alpha=10,
	n_rep_mcmc=5,
	n_rep_is=10,
	n_rep_tuning=20,
	n_rep_moments=10,
	n_pilot=10,
	true_value=0.0,
):
	"""Compare UPS, SUPS, and IS over sample sizes and target dimensions."""


	lambda_grid = jnp.asarray(lambda_grid)
	
	dimension_keys = jax.random.split(key, len(dimensions))
	results = {}
	for dimension, dimension_key in zip(dimensions, dimension_keys):
		key_tuning, key_moments, key_sigma, key_simulation = jax.random.split(
			dimension_key, 4
		)
		Sigma = jnp.eye(dimension)
		mean_init = -jnp.ones(dimension)
		Sigma_init = jnp.eye(dimension)
		D = alpha * jnp.ones(dimension)

		log_target_path = lambda x, path: _g_log_target(x, path, D)
		grad_log_target_path = lambda x, path: _g_grad_log_target(x, D)

		tuning = tune_km_grid(
			key_tuning,
			lambda_grid,
			Sigma,
			mean_init,
			Sigma_init,
			log_target_path,
			lag=1,
			nrep=n_rep_tuning,
		)
		k_grid = tuning["k_grid"]
		m_grid = tuning["m_grid"]
		lag = k_grid

		moment_results = estimate_moments(
			key_moments,
			lambda_grid,
			k_grid,
			m_grid,
			Sigma,
			mean_init,
			Sigma_init,
			log_target_path,
			grad_log_target_path,
			lag=1,
			nrep=n_rep_moments,
		)
		m2 = moment_results["m2"]
		sigma = estimate_sigma(
			key_sigma,
			lambda_grid,
			k_grid,
			m_grid,
			lag,
			mean_init,
			Sigma_init,
			Sigma,
			log_target_path,
			grad_log_target_path,
			n_pilot,
		)

		proposal = lambda sample_key: jax.random.normal(
			sample_key, shape=(dimension,)
		)
		log_weight = lambda x: _log_target1(x, D) - _log_target0(x)
		mcmc_key, allocation_key, is_key = jax.random.split(key_simulation, 3)
		replication_keys = jax.random.split(
			mcmc_key, len(M_values) * n_rep_mcmc
		)
		allocation_keys = jax.random.split(allocation_key, len(M_values))
		ups_mse_values = []
		sups_mse_values = []

		for i, sample_size in enumerate(tqdm(M_values, desc=f"d={dimension}")):
			ups_estimates = []
			sups_estimates = []
			budget = neyman_allocation(
				allocation_keys[i], lambda_grid, sigma, int(sample_size)
			)
			for rep in range(n_rep_mcmc):
				rep_key = replication_keys[i * n_rep_mcmc + rep]
				key_ups, key_sups = jax.random.split(rep_key)
				ups_estimate = UPS_estimator(
					key_ups,
					lambda_grid,
					m2,
					k_grid,
					m_grid,
					lag,
					mean_init,
					Sigma_init,
					Sigma,
					log_target_path,
					grad_log_target_path,
					M=int(sample_size),
				)
				ups_estimates.append(float(jax.device_get(ups_estimate)))

				sups_estimate = SUPS_estimator(
					key_sups,
					lambda_grid,
					budget,
					k_grid,
					m_grid,
					lag,
					mean_init,
					Sigma_init,
					Sigma,
					log_target_path,
					grad_log_target_path,
				)
				sups_estimates.append(float(jax.device_get(sups_estimate)))

			ups_mse_values.append(
				np.mean((np.asarray(ups_estimates) - true_value) ** 2)
			)
			sups_mse_values.append(
				np.mean((np.asarray(sups_estimates) - true_value) ** 2)
			)

		is_keys = jax.random.split(is_key, len(M_values))
		is_mse_values = []
		for sample_size, mse_key in zip(M_values, is_keys):
			is_replication_keys = jax.random.split(mse_key, n_rep_is)
			is_estimates = jax.vmap(
				lambda rep_key: IS_estimator(
					rep_key, log_weight, proposal, int(sample_size)
				)
			)(is_replication_keys)
			mse = jnp.mean((is_estimates - true_value) ** 2)
			is_mse_values.append(float(jax.device_get(mse)))

		results[dimension] = {
			"ups_mse": np.asarray(ups_mse_values),
			"sups_mse": np.asarray(sups_mse_values),
			"is_mse": np.asarray(is_mse_values),
		}

	sns.set_theme(style="whitegrid", context="notebook")
	estimator_styles = {
		"ups": ("UPS", "o", "#0072B2"),
		"sups": ("SUPS", "s", "#D55E00"),
		"is": ("IS", "^", "#009E73"),
	}
	figures = {}
	axes = {}
	for dimension in dimensions:
		fig, ax = plt.subplots(figsize=(7, 4.5))
		for estimator, (label, marker, color) in estimator_styles.items():
			values = results[dimension][f"{estimator}_mse"]
			ax.plot(
				M_values,
				M_values * values,
				marker=marker,
				markersize=7,
				linewidth=2.2,
				color=color,
				label=label,
			)
		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.set_xticks(M_values, labels=[str(value) for value in M_values])
		ax.set(
			title=f"UPS, SUPS and IS for dimension d={dimension}",
			xlabel=r"Sample size $M$",
			ylabel=r"$M \times \mathrm{MSE}$",
		)
		ax.legend(title="Estimator")
		sns.despine(ax=ax)
		fig.tight_layout()
		figures[dimension] = fig
		axes[dimension] = ax
		plt.show()

	return {
		"M_values": M_values,
		"results_by_dimension": results,
		"figures_by_dimension": figures,
		"axes_by_dimension": axes,
	}


def inefficiency_vs_alpha(
	key,
	alpha_values,
	dimension,
	M,
	lambda_grid,
	n_rep_mcmc=5,
	n_rep_is=10,
	n_rep_tuning=20,
	n_rep_moments=10,
	n_pilot=10,
	true_value=0.0,
):
	"""Compare M times MSE across mean shifts for one dimension and budget."""


	alpha_values = np.asarray(alpha_values, dtype=float)
	lambda_grid = jnp.asarray(lambda_grid)
	alpha_keys = jax.random.split(key, len(alpha_values))
	results = {"ups": [], "sups": [], "is": []}

	for alpha, alpha_key in zip(alpha_values, alpha_keys):
		key_tuning, key_moments, key_sigma, key_simulation = jax.random.split(
			alpha_key, 4
		)
		Sigma = jnp.eye(dimension)
		mean_init = -jnp.ones(dimension)
		Sigma_init = jnp.eye(dimension)
		D = alpha * jnp.ones(dimension)

		log_target_path = lambda x, path: _g_log_target(x, path, D)
		grad_log_target_path = lambda x, path: _g_grad_log_target(x, D)

		tuning = tune_km_grid(
			key_tuning,
			lambda_grid,
			Sigma,
			mean_init,
			Sigma_init,
			log_target_path,
			lag=1,
			nrep=n_rep_tuning,
		)
		k_grid = tuning["k_grid"]
		m_grid = tuning["m_grid"]
		lag = k_grid

		moment_results = estimate_moments(
			key_moments,
			lambda_grid,
			k_grid,
			m_grid,
			Sigma,
			mean_init,
			Sigma_init,
			log_target_path,
			grad_log_target_path,
			lag=1,
			nrep=n_rep_moments,
		)
		m2 = moment_results["m2"]
		sigma = estimate_sigma(
			key_sigma,
			lambda_grid,
			k_grid,
			m_grid,
			lag,
			mean_init,
			Sigma_init,
			Sigma,
			log_target_path,
			grad_log_target_path,
			n_pilot,
		)

		proposal = lambda sample_key: jax.random.normal(
			sample_key, shape=(dimension,)
		)

		log_weight = lambda x: _log_target1(x, D) - _log_target0(x)
		key_allocation, key_mcmc, key_is = jax.random.split(key_simulation, 3)
		budget = neyman_allocation(key_allocation, lambda_grid, sigma, M)
		replication_keys = jax.random.split(key_mcmc, n_rep_mcmc)

		ups_estimates = []
		sups_estimates = []
		for rep_key in replication_keys:
			key_ups, key_sups = jax.random.split(rep_key, 2)
			ups_estimates.append(
				float(
					jax.device_get(
						UPS_estimator(
							key_ups,
							lambda_grid,
							m2,
							k_grid,
							m_grid,
							lag,
							mean_init,
							Sigma_init,
							Sigma,
							log_target_path,
							grad_log_target_path,
							M=M,
						)
					)
				)
			)
			sups_estimates.append(
				float(
					jax.device_get(
						SUPS_estimator(
							key_sups,
							lambda_grid,
							budget,
							k_grid,
							m_grid,
							lag,
							mean_init,
							Sigma_init,
							Sigma,
							log_target_path,
							grad_log_target_path,
						)
					)
				)
			)

		is_keys = jax.random.split(key_is, n_rep_is)
		is_estimates = jax.vmap(
			lambda rep_key: IS_estimator(
				rep_key, log_weight, proposal, M
			)
		)(is_keys)
		is_estimates = np.asarray(jax.device_get(is_estimates))

		for estimator, estimates in (
			("ups", ups_estimates),
			("sups", sups_estimates),
			("is", is_estimates),
		):
			mse = np.mean((np.asarray(estimates) - true_value) ** 2)
			results[estimator].append({"mse": mse, "M_times_mse": M * mse})

	results = {
		estimator: {
			"mse": np.asarray([point["mse"] for point in values]),
			"M_times_mse": np.asarray(
				[point["M_times_mse"] for point in values]
			),
		}
		for estimator, values in results.items()
	}

	sns.set_theme(style="whitegrid", context="notebook")
	fig, ax = plt.subplots(figsize=(7, 4.5))
	estimator_styles = {
		"ups": ("UPS", "o", "#0072B2"),
		"sups": ("SUPS", "s", "#D55E00"),
		"is": ("IS", "^", "#009E73"),
	}
	for estimator, (label, marker, color) in estimator_styles.items():
		ax.plot(
			alpha_values,
			results[estimator]["M_times_mse"],
			marker=marker,
			markersize=7,
			linewidth=2.2,
			color=color,
			label=label,
		)
	ax.set(
		title=f"Estimator inefficiency versus alpha (d={dimension}, M={M})",
		xlabel=r"Mean shift $\alpha$",
		ylabel=r"$M \times \mathrm{MSE}$",
	)
	ax.legend(title="Estimator")
	sns.despine(ax=ax)
	fig.tight_layout()
	plt.show()

	return {
		"alpha_values": alpha_values,
		"dimension": dimension,
		"M": M,
		"results_by_estimator": results,
		"figure": fig,
		"axes": ax,
	}


def plot_estimator_boxplot(
	key,
	dimension,
	M,
	lambda_grid,
	alpha=10,
	n_rep_mcmc=20,
	n_rep_is=20,
	n_rep_tuning=20,
	n_rep_moments=10,
	n_pilot=10,
	true_value=0.0,
):
	"""Plot UPS, SUPS, and IS estimate distributions for one setup."""

	lambda_grid = jnp.asarray(lambda_grid)
	(
		key_tuning,
		key_moments,
		key_sigma,
		key_allocation,
		key_ups,
		key_sups,
		key_is,
	) = jax.random.split(key, 7)

	Sigma = jnp.eye(dimension)
	mean_init = -jnp.ones(dimension)
	Sigma_init = jnp.eye(dimension)
	D = alpha * jnp.ones(dimension)
	log_target_path = lambda x, path: _g_log_target(x, path, D)
	grad_log_target_path = lambda x, path: _g_grad_log_target(x, D)

	tuning = tune_km_grid(
		key_tuning,
		lambda_grid,
		Sigma,
		mean_init,
		Sigma_init,
		log_target_path,
		lag=1,
		nrep=n_rep_tuning,
	)
	k_grid = tuning["k_grid"]
	m_grid = tuning["m_grid"]
	lag = k_grid
	moment_results = estimate_moments(
		key_moments,
		lambda_grid,
		k_grid,
		m_grid,
		Sigma,
		mean_init,
		Sigma_init,
		log_target_path,
		grad_log_target_path,
		lag=1,
		nrep=n_rep_moments,
	)
	m2 = moment_results["m2"]
	sigma = estimate_sigma(
		key_sigma,
		lambda_grid,
		k_grid,
		m_grid,
		lag,
		mean_init,
		Sigma_init,
		Sigma,
		log_target_path,
		grad_log_target_path,
		n_pilot,
	)
	budget = neyman_allocation(key_allocation, lambda_grid, sigma, M)

	ups_keys = jax.random.split(key_ups, n_rep_mcmc)
	sups_keys = jax.random.split(key_sups, n_rep_mcmc)
	ups_estimates = [
		float(
			jax.device_get(
				UPS_estimator(
					rep_key,
					lambda_grid,
					m2,
					k_grid,
					m_grid,
					lag,
					mean_init,
					Sigma_init,
					Sigma,
					log_target_path,
					grad_log_target_path,
					M=M,
				)
			)
		)
		for rep_key in ups_keys
	]
	sups_estimates = [
		float(
			jax.device_get(
				SUPS_estimator(
					rep_key,
					lambda_grid,
					budget,
					k_grid,
					m_grid,
					lag,
					mean_init,
					Sigma_init,
					Sigma,
					log_target_path,
					grad_log_target_path,
				)
			)
		)
		for rep_key in sups_keys
	]
	proposal = lambda sample_key: jax.random.normal(
		sample_key, shape=(dimension,)
	)
	log_weight = lambda x: _log_target1(x, D) - _log_target0(x)
	is_keys = jax.random.split(key_is, n_rep_is)
	is_estimates = jax.vmap(
		lambda rep_key: IS_estimator(
			rep_key, log_weight, proposal, M = 1000
		)
	)(is_keys)

	estimates = {
		"UPS": np.asarray(ups_estimates),
		"SUPS": np.asarray(sups_estimates),
		"IS": np.asarray(jax.device_get(is_estimates)),
	}
	sns.set_theme(style="whitegrid", context="notebook")
	fig, ax = plt.subplots(figsize=(7, 4.5))
	boxplot = ax.boxplot(
		list(estimates.values()),
		patch_artist=True,
		showmeans=True,
	)
	for patch, color in zip(
		boxplot["boxes"], ("#0072B2", "#D55E00", "#009E73")
	):
		patch.set_facecolor(color)
		patch.set_alpha(0.65)
	ax.set_xticks([1, 2, 3], labels=list(estimates))
	true_value_line = ax.axhline(
		true_value,
		color="0.35",
		linestyle="--",
		linewidth=1.2,
		label="True value",
	)
	ax.set(
		ylabel="Estimate of log(Z1 / Z0)",
	)
	legend_handles = [
		Patch(facecolor=color, alpha=0.65, label=label)
		for label, color in zip(
			("UPS", "SUPS", "IS"), ("#0072B2", "#D55E00", "#009E73")
		)
	]
	ax.legend(handles=legend_handles + [true_value_line])
	sns.despine(ax=ax)
	fig.tight_layout()
	plt.show()

	return {
		"estimates_by_estimator": estimates,
		"figure": fig,
		"axes": ax,
	}
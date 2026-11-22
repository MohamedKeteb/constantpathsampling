import numpy as np

from unbiased_mcmc.MH_coupling import inv_cholesky
from unbiased_mcmc.UPS import E_lambda


def estimate_sigma(
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
	N_pilot,
):
	lambda_grid = np.asarray(lambda_grid)
	bin_count = lambda_grid.size - 1
	variances = []

	for bin_index in range(bin_count):
		lambdas = np.random.uniform(
			lambda_grid[bin_index],
			lambda_grid[bin_index + 1],
		size=N_pilot,
		)
		E_values = np.asarray(
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
				for lam in lambdas
			]
		)
		variances.append(np.var(E_values, ddof=1))

	return np.sqrt(np.asarray(variances))



def neyman_allocation(lambda_grid, sigma, M):
	lambda_grid = np.asarray(lambda_grid)
	sigma = np.asarray(sigma)
	bin_count = lambda_grid.size - 1
	M = int(M)
	if M < bin_count:
		raise ValueError("M must be at least the number of bins")

	widths = np.diff(lambda_grid)
	scores = widths * sigma
	score_sum = np.sum(scores)
	if score_sum <= 0 or not np.isfinite(score_sum):
		probs = np.full(bin_count, 1.0 / bin_count)
	else:
		probs = scores / score_sum

	remaining_budget = M - bin_count
	ideal = remaining_budget * probs
	base = np.floor(ideal).astype(int)
	residual = ideal - base
	residual_count = remaining_budget - np.sum(base)

	if residual_count > 0:
		residual_sum = np.sum(residual)
		residual_probs = probs if residual_sum <= 0 else residual / residual_sum
		residual_counts = np.random.multinomial(residual_count, residual_probs)
	else:
		residual_counts = np.zeros(bin_count, dtype=int)

	return 1 + base + residual_counts


def SUPS_estimator(
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
):
	lambda_grid = np.asarray(lambda_grid)
	budget = np.asarray(budget, dtype=int)
	widths = np.diff(lambda_grid)

	estimates_per_bin = []
	costs_per_bin = []
	for bin_index, sample_count in enumerate(budget):
		lambdas = np.random.uniform(
			lambda_grid[bin_index],
			lambda_grid[bin_index + 1],
			size=int(sample_count),
		)
		E_lambda_results = [
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
			)
			for lam in lambdas
		]
		E_values = np.asarray([result[0] for result in E_lambda_results])
		costs_per_bin.append(np.sum([result[1] for result in E_lambda_results]))
		estimates_per_bin.append(widths[bin_index] * np.mean(E_values, axis=0))

	estimate = np.sum(np.stack(estimates_per_bin), axis=0)
	total_cost = np.sum(costs_per_bin)
	return estimate, total_cost

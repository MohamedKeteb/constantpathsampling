import numpy as np

from unbiased_mcmc.MH_coupling import (
	MH_coupled_kernel,
	MH_kernel,
	initial_distribution,
	inv_cholesky,
)
from unbiased_mcmc.umcmc import unbiased_estimator


def E_lambda(
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
):
	lambda_grid = np.asarray(lambda_grid)
	index_lambda = np.argmin(np.abs(lambda_grid - lam))
	k_lam = int(k_grid[index_lambda])
	m_lam = int(m_grid[index_lambda])
	lag_array = np.asarray(lag)
	lag_lam = int(lag_array) if lag_array.ndim == 0 else int(lag_array[index_lambda])

	log_target = lambda x: log_target_path(x, path=lam)
	grad_log_target = lambda x: grad_log_target_path(x, path=lam)

	def initial():
		return initial_distribution(log_target, mean_init, Sigma_init)

	def kernel(state, log_pdf):
		return MH_kernel(
			state, log_pdf, Sigma, cholesky_Sigma, log_target
		)

	def coupled_kernel(state1, pdf1, state2, pdf2):
		return MH_coupled_kernel(
			state1,
			pdf1,
			state2,
			pdf2,
			cholesky_Sigma,
			inv_cholesky_Sigma,
			log_target
		)

	umcmc_estimator = unbiased_estimator(
		kernel,
		coupled_kernel,
		initial,
		h=[grad_log_target],
		k=k_lam,
		m=m_lam,
		lag=lag_lam,
	)

	return umcmc_estimator["uestimator"][0], umcmc_estimator["cost"]


def optimal_q(lambda_grid, m2):
	lambda_grid = np.asarray(lambda_grid)
	m2 = np.asarray(m2)

	sqrt_m2 = np.sqrt(np.maximum(m2, 0.0))
	widths = np.diff(lambda_grid)
	heights = 0.5 * (sqrt_m2[:-1] + sqrt_m2[1:])
	weights = widths * heights
	weights = weights / np.sum(weights)
	cdf = np.cumsum(weights)

	u, v = np.random.uniform(size=2)
	index = min(np.searchsorted(cdf, u, side="right"), lambda_grid.size - 2)
	lam = lambda_grid[index] + v * widths[index]
	q_lam = weights[index] / widths[index]
	return lam, q_lam


def UPS_estimator(
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
):

	lambda_samples = [optimal_q(lambda_grid, m2) for _ in range(M)]
	lambdas = np.asarray([sample[0] for sample in lambda_samples])
	q_lambdas = np.asarray([sample[1] for sample in lambda_samples])

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
	E_lambda_estimates = np.asarray([result[0] for result in E_lambda_results])
	total_cost = np.sum([result[1] for result in E_lambda_results])
	importance_densities = q_lambdas.reshape(
		(q_lambdas.size,) + (1,) * (E_lambda_estimates.ndim - 1)
	)
	estimates = E_lambda_estimates / importance_densities

	return np.mean(estimates), total_cost

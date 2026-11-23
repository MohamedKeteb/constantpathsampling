import numpy as np
import pandas as pd
import plotnine as p9

from unbiased_mcmc.MH_coupling import (
	MH_coupled_kernel,
	MH_kernel,
	cholesky,
	initial_distribution,
	inv_cholesky,
)
from unbiased_mcmc.UPS import E_lambda
from unbiased_mcmc.umcmc import coupled_chain, unbiased_estimator


def tune_km_grid(
	lambda_grid,
	mean_init,
	Sigma_init,
	Sigma,
	cholesky_Sigma,
	inv_cholesky_Sigma,
	log_target_path,
	lag=1,
	nrep=100,
):
	lambda_grid = np.asarray(lambda_grid)
	k_grid = []
	meantau_grid = []
	meetings = []

	for lambda_index, lam in enumerate(lambda_grid):
		log_target = lambda x, lam=lam: log_target_path(x, path=lam)

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
				log_target,
			)

		meeting_times = np.asarray(
			[
				coupled_chain(kernel, coupled_kernel, initial, k=0, lag=lag)[
					"meetingtime"
				]
				for _ in range(nrep)
			]
		)
		k_grid.append(int(np.floor(np.quantile(meeting_times, 0.99))))
		meantau_grid.append(np.mean(meeting_times))
		meetings.append(
			pd.DataFrame(
				{
					"ilambda": lambda_index,
					"lambda": float(lam),
					"meetings": meeting_times,
				}
			)
		)

	k_grid = np.asarray(k_grid, dtype=int)
	meantau = np.asarray(meantau_grid)
	m_grid = (
		5 * np.max(k_grid)
		+ np.floor(np.max(meantau))
		- np.floor(meantau)
	).astype(int)
	meetings_df = pd.concat(meetings, ignore_index=True)

	return {
		"k_grid": k_grid,
		"m_grid": m_grid,
		"meetings_df": meetings_df,
		"meantau": meantau,
	}


def plot_km_grid(results, lambda_grid):
	lambda_values = np.asarray(lambda_grid)
	meetings_df = results["meetings_df"].copy()
	k_df = pd.DataFrame(
		{"lambda": lambda_values, "k": np.asarray(results["k_grid"])}
	)
	k_df["series"] = "k"
	violin_width = 0.8 * np.min(np.diff(lambda_values))

	return (
		p9.ggplot()
		+ p9.geom_violin(
			data=meetings_df,
			mapping=p9.aes(x="lambda", y="meetings", group="lambda"),
			width=violin_width,
			fill="white",
			color="black",
			size=0.35,
		)
		+ p9.geom_line(
			data=k_df,
			mapping=p9.aes(x="lambda", y="k", color="series"),
			inherit_aes=False,
			size=0.6,
		)
		+ p9.geom_point(
			data=k_df,
			mapping=p9.aes(x="lambda", y="k", color="series"),
			inherit_aes=False,
			fill="white",
			size=1.5,
		)
		+ p9.scale_color_manual(
			values={"k": "black"},
			labels={"k": r"$k_\lambda$"},
		)
		+ p9.scale_x_continuous(
			breaks=np.linspace(lambda_values.min(), lambda_values.max(), 5)
		)
		+ p9.labs(x=r"$\lambda$", y="Meeting time", color="")
		+ p9.theme_classic(base_size=9)
		+ p9.theme(
			figure_size=(5, 3),
			legend_position=(0.02, 0.98),
			legend_justification=(0, 1),
			legend_direction="horizontal",
			legend_text=p9.element_text(size=15),
			plot_background=p9.element_rect(fill="white", color="white"),
			panel_background=p9.element_rect(fill="white", color="white"),
		)
	)


def estimate_moments(
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
	lag=1,
	nrep=20,
):
	lambda_grid = np.asarray(lambda_grid)
	m1_values = []
	m2_values = []
	variance_values = []

	for lambda_index, lam in enumerate(lambda_grid):
		estimates = np.asarray(
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
		mean_estimate = np.mean(estimates, axis=0)
		flat_estimates = estimates.reshape(nrep, -1)
		flat_mean = np.asarray(mean_estimate).reshape(-1)
		m1_values.append(mean_estimate)
		m2_values.append(np.mean(np.sum(flat_estimates**2, axis=1)))
		variance_values.append(
			np.mean(np.sum((flat_estimates - flat_mean) ** 2, axis=1))
		)

	return {
		"m1": np.asarray(m1_values),
		"m2": np.asarray(m2_values),
		"var": np.asarray(variance_values),
	}


def plot_m1_m2(results, lambda_grid):
	lambda_values = np.asarray(lambda_grid)
	m1 = np.asarray(results["m1"])
	m2 = np.asarray(results["m2"])
	data = pd.concat(
		[
			pd.DataFrame(
				{
					"lambda": lambda_values,
					"Quantity": r"$\sqrt{\widehat{m}_2(\lambda)}$",
					"Value": np.sqrt(np.maximum(m2, 0.0)),
				}
			),
			pd.DataFrame(
				{
					"lambda": lambda_values,
					"Quantity": r"$\widehat{m}_1(\lambda)$",
					"Value": m1,
				}
			),
		],
		ignore_index=True,
	)

	return (
		p9.ggplot(data, p9.aes(x="lambda", y="Value", group="Quantity"))
		+ p9.geom_line(color="black", size=0.6)
		+ p9.geom_point(color="black", fill="white", size=1.5)
		+ p9.facet_wrap("~Quantity", ncol=1, scales="free_y")
		+ p9.labs(x=r"$\lambda$", y="")
		+ p9.theme_classic(base_size=15)
		+ p9.theme(
			figure_size=(5, 3),
			strip_background=p9.element_blank(),
			strip_text=p9.element_text(size=15),
			axis_title_x=p9.element_text(size=15),
			axis_title_y=p9.element_blank(),
			panel_background=p9.element_rect(fill="white", color="white"),
			plot_background=p9.element_rect(fill="white", color="white"),
		)
	)


def tune_initial_distribution(
	lambda_grid,
	k_grid,
	m_grid,
	Sigma,
	mean_init,
	Sigma_init,
	log_target_path,
	lag=1,
):
	lambda_grid = np.asarray(lambda_grid)
	mean_init = np.asarray(mean_init)
	cholesky_sigma = cholesky(Sigma)
	inv_cholesky_sigma = inv_cholesky(Sigma)
	lambda_count = lambda_grid.size
	dimension = mean_init.size
	m1 = np.zeros((lambda_count, dimension))
	m2 = np.zeros((lambda_count, dimension, dimension))
	covariance = np.zeros((lambda_count, dimension, dimension))

	for lambda_index, lam in enumerate(lambda_grid):
		log_target = lambda x, lam=lam: log_target_path(x, path=lam)

		def initial():
			return initial_distribution(log_target, mean_init, Sigma_init)

		def kernel(state, log_pdf):
			return MH_kernel(
				state, log_pdf, Sigma, cholesky_sigma, log_target
			)

		def coupled_kernel(state1, pdf1, state2, pdf2):
			return MH_coupled_kernel(
				state1,
				pdf1,
				state2,
				pdf2,
				cholesky_sigma,
				inv_cholesky_sigma,
				log_target,
			)

		res = unbiased_estimator(
			kernel,
			coupled_kernel,
			initial,
			h=[lambda x: x, lambda x: np.outer(x, x)],
			k=int(k_grid[lambda_index]),
			m=int(m_grid[lambda_index]),
			lag=lag,
		)
		m1[lambda_index] = res["uestimator"][0]
		m2[lambda_index] = res["uestimator"][1]
		covariance[lambda_index] = m2[lambda_index] - np.outer(
			m1[lambda_index], m1[lambda_index]
		)

	return {"m1": m1, "m2": m2, "covariance": covariance}

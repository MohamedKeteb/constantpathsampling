import numpy as np


def _proposal_from_cholesky(current_state, cholesky_Sigma):
	standard_noise = np.random.normal(size=current_state.shape)
	if cholesky_Sigma.ndim == 1:
		noise = cholesky_Sigma * standard_noise
	else:
		noise = cholesky_Sigma @ standard_noise
	return current_state + noise


def _metropolis_accept(
	current_state, current_log_pdf, proposal, proposal_log_pdf
):
	accept = np.log(np.random.uniform()) < proposal_log_pdf - current_log_pdf
	new_state = np.where(accept, proposal, current_state)
	new_log_pdf = np.where(accept, proposal_log_pdf, current_log_pdf)
	return new_state, new_log_pdf


def _metropolis_accept_pair(
	states, log_pdfs, proposals, proposal_log_pdfs
):
	log_uniforms = np.log(np.random.uniform(size=2))
	accept = log_uniforms < proposal_log_pdfs - log_pdfs
	new_states = np.where(accept[:, None], proposals, states)
	new_log_pdfs = np.where(accept, proposal_log_pdfs, log_pdfs)
	return new_states, new_log_pdfs


def MH_kernel(
	current_state,
	current_log_pdf,
	Sigma,
	cholesky_Sigma,
	log_target
):
	current_state = np.asarray(current_state)

	proposal = _proposal_from_cholesky(
			current_state, np.asarray(cholesky_Sigma)
		)

	proposal_log_pdf = log_target(proposal)
	return _metropolis_accept(
		current_state, current_log_pdf, proposal, proposal_log_pdf
	)


def initial_distribution(log_target, mean_init, Sigma_init):
	current_state = np.random.multivariate_normal(mean=mean_init, cov=Sigma_init)
	current_log_pdf = log_target(current_state)
	return current_state, current_log_pdf


def cholesky(Sigma):
	Sigma = np.asarray(Sigma)
	if np.allclose(Sigma, np.diag(np.diag(Sigma))):
		return np.sqrt(np.diag(Sigma))
	return np.linalg.cholesky(Sigma)


def inv_cholesky(Sigma):
	cholesky_sigma = cholesky(Sigma)
	if cholesky_sigma.ndim == 1:
		return 1.0 / cholesky_sigma
	return np.linalg.inv(cholesky_sigma)


def reflection_maximal_coupling(
	mu1, mu2, cholesky_Sigma, inv_cholesky_Sigma
):
	mu1 = np.asarray(mu1)
	mu2 = np.asarray(mu2)
	cholesky_Sigma = np.asarray(cholesky_Sigma)
	standard_normal = np.random.normal(size=mu1.shape)
	log_uniform = np.log(np.random.uniform())

	if cholesky_Sigma.ndim == 1:
		delta = inv_cholesky_Sigma * (mu1 - mu2)
	else:
		delta = inv_cholesky_Sigma @ (mu1 - mu2)

	direction = delta / max(np.linalg.norm(delta), np.finfo(mu1.dtype).tiny)
	log_ratio = (
		-0.5 * np.sum((standard_normal + delta) ** 2)
		+ 0.5 * np.sum(standard_normal**2)
	)
	if log_uniform <= log_ratio:
		reflected_normal = standard_normal + delta
	else:
		reflected_normal = (
			standard_normal
			- 2 * np.dot(direction, standard_normal) * direction
		)

	if cholesky_Sigma.ndim == 1:
		proposal1 = mu1 + cholesky_Sigma * standard_normal
		proposal2 = mu2 + cholesky_Sigma * reflected_normal
	else:
		proposal1 = mu1 + cholesky_Sigma @ standard_normal
		proposal2 = mu2 + cholesky_Sigma @ reflected_normal
	return proposal1, proposal2


def MH_coupled_kernel(
	current_state1,
	current_pdf1,
	current_state2,
	current_pdf2,
	cholesky_Sigma,
	inv_cholesky_Sigma,
	log_target
):
	proposal1, proposal2 = reflection_maximal_coupling(
		current_state1,
		current_state2,
		cholesky_Sigma,
		inv_cholesky_Sigma,
	)
	proposal_pdfs = np.asarray(
		[log_target(proposal1), log_target(proposal2)]
	)
	states = np.stack((current_state1, current_state2))
	log_pdfs = np.asarray([current_pdf1, current_pdf2])
	proposals = np.stack((proposal1, proposal2))
	new_states, new_pdfs = _metropolis_accept_pair(
		states, log_pdfs, proposals, proposal_pdfs
	)
	return new_states[0], new_pdfs[0], new_states[1], new_pdfs[1]

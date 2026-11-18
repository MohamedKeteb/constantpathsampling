import jax
import jax.numpy as jnp
import jax.random as random


@jax.jit
def _proposal_from_cholesky(key, current_state, cholesky_sigma):
    standard_noise = random.normal(
        key, shape=current_state.shape, dtype=current_state.dtype
    )
    if cholesky_sigma.ndim == 1:
        noise = cholesky_sigma * standard_noise
    else:
        noise = cholesky_sigma @ standard_noise
    return current_state + noise


@jax.jit
def _metropolis_accept(key, current_state, current_log_pdf, proposal, proposal_log_pdf):
    log_u = jnp.log(random.uniform(key))
    accept = log_u < proposal_log_pdf - current_log_pdf
    new_state = jnp.where(accept, proposal, current_state)
    new_log_pdf = jnp.where(accept, proposal_log_pdf, current_log_pdf)
    return new_state, new_log_pdf


@jax.jit
def _metropolis_accept_pair(keys, states, log_pdfs, proposals, proposal_log_pdfs):
    return jax.vmap(_metropolis_accept)(
        keys, states, log_pdfs, proposals, proposal_log_pdfs
    )


# RWMH kernel with a gaussian proposal of covaraince Sigma

def MH_kernel(key, current_state, current_log_pdf, Sigma, log_target, cholesky_sigma=None):

    d = current_state.shape[0]
    key, key_proposal, key_accept = random.split(key, 3)

    if cholesky_sigma is None:
        noise = random.multivariate_normal(
            key_proposal, mean=jnp.zeros(d), cov=Sigma
        )
        proposal = current_state + noise
    else:
        proposal = _proposal_from_cholesky(
            key_proposal, current_state, cholesky_sigma
        )

    proposal_log_pdf = log_target(proposal)
    new_state, new_log_pdf = _metropolis_accept(
        key_accept, current_state, current_log_pdf, proposal, proposal_log_pdf
    )
    return key, new_state, new_log_pdf

# Gaussian Initial distribution 


def initial_distribution(key, log_target, mean_init, Sigma_init):

    key, subkey = random.split(key)

    current_state = random.multivariate_normal(subkey, mean=mean_init, cov=Sigma_init)
    current_log_pdf = log_target(current_state)

    return key, current_state, current_log_pdf

# Implementation of the reflection maximal coupling between two multivaraite normal distribution with the same covaraince 

# cholesky_Sigma =  jnp.linalg.cholesky(Sigma)


def cholesky(Sigma):
    if jnp.allclose(Sigma, jnp.diag(jnp.diag(Sigma))):
        return jnp.sqrt(jnp.diag(Sigma))
    else:
        return jnp.linalg.cholesky(Sigma)

@jax.jit
def reflection_maximal_coupling(key, mu1, mu2, cholesky_Sigma):

    d = mu1.shape[0]
    key, key_x, key_u = random.split(key, 3)

    x_dot = random.normal(key_x, shape=(d,))
    log_u = jnp.log(random.uniform(key_u))

    if cholesky_Sigma.ndim == 1:
        # Cas diagonal
        Delta = (mu1 - mu2) / cholesky_Sigma

        e = Delta / jnp.maximum(jnp.linalg.norm(Delta),jnp.finfo(mu1.dtype).tiny)

        log_ratio = -0.5 * jnp.sum((x_dot + Delta)**2) + 0.5 * jnp.sum(x_dot**2)

        y_dot = jnp.where(log_u <= log_ratio,x_dot + Delta,x_dot - 2 * jnp.dot(e, x_dot) * e)

        x = mu1 + cholesky_Sigma * x_dot
        y = mu2 + cholesky_Sigma * y_dot

    else:
        # Cas dense
        Delta = jax.scipy.linalg.solve_triangular(cholesky_Sigma, mu1 - mu2, lower=True)

        e = Delta / jnp.maximum(jnp.linalg.norm(Delta),jnp.finfo(mu1.dtype).tiny)

        log_ratio = -0.5 * jnp.sum((x_dot + Delta)**2) + 0.5 * jnp.sum(x_dot**2)

        y_dot = jnp.where(log_u <= log_ratio,x_dot + Delta,x_dot - 2 * jnp.dot(e, x_dot) * e)

        x = mu1 + cholesky_Sigma @ x_dot
        y = mu2 + cholesky_Sigma @ y_dot

    return key, x, y

# Coupled RWMH with reflection maximal coupling

def MH_coupled_kernel(key, current_state1, current_pdf1, current_state2, current_pdf2, cholesky_sigma, log_target):

    key, key_coupling, key_u1, key_u2 = random.split(key, 4)

    _, proposal1, proposal2 = reflection_maximal_coupling(key_coupling, current_state1, current_state2, cholesky_sigma)

    proposal_pdfs = jnp.stack((log_target(proposal1), log_target(proposal2)))
    new_states, new_pdfs = _metropolis_accept_pair(
        jnp.stack((key_u1, key_u2)),
        jnp.stack((current_state1, current_state2)),
        jnp.stack((current_pdf1, current_pdf2)),
        jnp.stack((proposal1, proposal2)),
        proposal_pdfs,
    )

    return key, new_states[0], new_pdfs[0], new_states[1], new_pdfs[1]


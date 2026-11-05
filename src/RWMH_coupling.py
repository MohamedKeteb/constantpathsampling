import jax
import jax.numpy as jnp
import jax.random as random

# RWMH kernel with a gaussian proposal of covaraince Sigma

def MH_kernel(key, current_state, current_log_pdf, Sigma, log_target):

    d = current_state.shape[0]
    key, key_proposal, key_accept = random.split(key, 3)

    noise = random.multivariate_normal(key_proposal, mean=jnp.zeros(d), cov=Sigma)
    proposal = current_state + noise

    proposal_log_pdf = log_target(proposal)

    log_alpha = proposal_log_pdf - current_log_pdf
    log_u = jnp.log(random.uniform(key_accept))

    accept = log_u < log_alpha

    new_state = jnp.where(accept, proposal, current_state)
    new_log_pdf = jnp.where(accept, proposal_log_pdf, current_log_pdf)

    return key, new_state, new_log_pdf

# Gaussian Initial distribution 


def initial_distribution(key, log_target, mean_init, Sigma_init):

    key, subkey = random.split(key)

    current_state = random.multivariate_normal(subkey, mean=mean_init, cov=Sigma_init)
    current_log_pdf = log_target(current_state)

    return key, current_state, current_log_pdf

# Implementation of the reflection maximal coupling between two multivaraite normal distribution with the same covaraince 


def reflection_maximal_coupling(key, mu1, mu2, Sigma):

    d = mu1.shape[0]
    key, key_x, key_u = random.split(key, 3)

    L = jnp.linalg.cholesky(Sigma)

    Delta = jnp.linalg.solve(L, mu1 - mu2)
    e = Delta / jnp.maximum(jnp.linalg.norm(Delta), jnp.finfo(mu1.dtype).tiny)

    x_dot = random.normal(key_x, shape=(d,))
    log_u = jnp.log(random.uniform(key_u))

    log_ratio = -0.5 * jnp.sum((x_dot + Delta)**2) + 0.5 * jnp.sum(x_dot**2)

    y_dot = jnp.where(log_u <= log_ratio, x_dot + Delta, x_dot - 2 * jnp.dot(e, x_dot) * e)

    x = mu1 + L @ x_dot
    y = mu2 + L @ y_dot

    return key, x, y


# Coupled RWMH with reflection maximal coupling

def MH_coupled_kernel(key, current_state1, current_pdf1, current_state2, current_pdf2, Sigma, log_target):

    key, key_coupling, key_u1, key_u2 = random.split(key, 4)

    _, proposal1, proposal2 = reflection_maximal_coupling(key_coupling, current_state1, current_state2, Sigma)

    proposal_pdf1 = log_target(proposal1)
    proposal_pdf2 = log_target(proposal2)

    accept1 = jnp.log(random.uniform(key_u1)) < proposal_pdf1 - current_pdf1
    accept2 = jnp.log(random.uniform(key_u2)) < proposal_pdf2 - current_pdf2

    new_state1 = jnp.where(accept1, proposal1, current_state1)
    new_pdf1 = jnp.where(accept1, proposal_pdf1, current_pdf1)

    new_state2 = jnp.where(accept2, proposal2, current_state2)
    new_pdf2 = jnp.where(accept2, proposal_pdf2, current_pdf2)

    return key, new_state1, new_pdf1, new_state2, new_pdf2


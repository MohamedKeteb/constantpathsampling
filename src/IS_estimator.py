# Here we implement the importance sampling estimator by doing importance sampling from pi_0 to pi_1
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random
from jax.scipy.special import logsumexp


@partial(jax.jit, static_argnames=("log_weight", "proposal", "M"))
def IS_estimator(key, log_weight, proposal, M):
    keys = jax.random.split(key, M)

    samples = jax.vmap(proposal)(keys)

    log_weights = jax.vmap(log_weight)(samples)

    return logsumexp(log_weights) - jnp.log(M)



    

import jax
import jax.numpy as jnp
import jax.random as random
import math
from tqdm import tqdm

# Definition of the lag coupling as in Biswas et al. (2018) and Jacob et al. (2020)

def coupled_chain(key, kernel, coupled_kernel, initial, k, lag, max_iterations=float("inf"), preallocate=100):

    # Initialization
    key, key_init1, key_init2 = random.split(key, 3)
    _, initial_state1, initial_pdf1 = initial(key_init1)
    _, initial_state2, initial_pdf2 = initial(key_init2)

    current_state1, current_pdf1 = initial_state1, initial_pdf1
    current_state2, current_pdf2 = initial_state2, initial_pdf2

    current_state1 = jnp.asarray(current_state1)
    current_state2 = jnp.asarray(current_state2)

    d = current_state1.size

    # Initial storage
    nrowsamples1 = k + preallocate + lag
    samples1 = jnp.full((nrowsamples1, d), jnp.nan)
    samples2 = jnp.full((k + preallocate, d), jnp.nan)

    samples1 = samples1.at[0].set(current_state1)
    samples2 = samples2.at[0].set(current_state2)

    current_nsamples1 = 1
    iteration = 0

    # First chain runs during the lag
    for _ in range(lag):

        iteration += 1
        key, current_state1, current_pdf1 = kernel(key, current_state1, current_pdf1)
        samples1 = samples1.at[current_nsamples1].set(current_state1)
        current_nsamples1 += 1

    # Coupled simulation
    meet = False
    finished = False
    meeting_time = float("inf")

    while not finished and iteration < max_iterations:
        iteration += 1
        if meet:

            key, current_state1, current_pdf1 = kernel(key, current_state1, current_pdf1)
            current_state2 = current_state1
            current_pdf2 = current_pdf1

        else:

            (
                key,
                current_state1,
                current_pdf1,
                current_state2,
                current_pdf2
            ) = coupled_kernel(
                key,
                current_state1,
                current_pdf1,
                current_state2,
                current_pdf2
            )

            if jnp.array_equal(current_state1, current_state2):
                meet = True
                meeting_time = iteration

        # Dynamic memory allocation
        if current_nsamples1 >= nrowsamples1:

            samples1 = jnp.concatenate((samples1, jnp.full((nrowsamples1, d), jnp.nan)), axis=0)
            samples2 = jnp.concatenate((samples2, jnp.full((nrowsamples1, d), jnp.nan)), axis=0)
            nrowsamples1 *= 2

        # Store samples
        samples1 = samples1.at[current_nsamples1].set(current_state1)
        samples2 = samples2.at[current_nsamples1 - lag].set(current_state2)

        current_nsamples1 += 1

        # Stopping condition
        if iteration >= max(meeting_time, k):
            finished = True

    # Remove unused memory
    samples1 = samples1[:iteration + 1]
    samples2 = samples2[:iteration - lag + 1]

    return {"samples1": samples1, "samples2": samples2, "meetingtime": meeting_time, "iteration": iteration, "finished": finished, "key": key}



# Compute the unbiased MCMC as in Jacob et al. (2020)


def H_bar(c_chain, h_list, k, m, lag):

    samples1 = c_chain["samples1"]
    samples2 = c_chain["samples2"]
    meeting_time = c_chain["meetingtime"]
    maxiter = c_chain["iteration"]

    # -------- Checks --------
    if k > maxiter or m > maxiter:
        print("error: k and m must be <= maxiter")
        return None

    # -------- Main term --------
    H_bar_val = []

    for h in h_list:
        h_of_chain = jnp.stack([jnp.asarray(h(x)) for x in samples1[k:m+1, :]])
        H_bar_val.append(jnp.sum(h_of_chain, axis=0))

    # -------- Correction term --------
    deltas_term = []

    for h in h_list:
        correction = 0.0

        if meeting_time > k + lag:
            for t in range(k + lag, meeting_time):
                coefficient = math.floor((t-k)/lag) - math.ceil(max(lag, t-m)/lag) + 1
                delta = jnp.squeeze(h(samples1[t, :])) - jnp.squeeze(h(samples2[t-lag, :]))
                correction = correction + coefficient * delta

        deltas_term.append(correction)

    return [
        (main_term + correction_term) / (m - k + 1)
        for main_term, correction_term in zip(H_bar_val, deltas_term)
    ]

# Combine the coupled chain and H_bar to compute the unbiased estimator

def unbiased_estimator(key, kernel, coupled_kernel, initial, h, k, m, lag):
    # Coupled chain
    c_chain = coupled_chain(key, kernel, coupled_kernel, initial, m, lag)

    # Compute H_bar
    H_bar_val = H_bar(c_chain, h, k, m, lag)

    # Add the estimator to the dictionary
    c_chain["uestimator"] = H_bar_val

    return c_chain



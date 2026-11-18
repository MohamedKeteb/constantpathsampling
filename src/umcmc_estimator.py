
import jax
import jax.numpy as jnp
import jax.random as random
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

    samples1 = [current_state1]
    samples2 = [current_state2]

    current_nsamples1 = 1
    iteration = 0

    # First chain runs during the lag
    for _ in range(lag):

        iteration += 1
        key, current_state1, current_pdf1 = kernel(key, current_state1, current_pdf1)
        samples1.append(current_state1)
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

        samples1.append(current_state1)
        samples2.append(current_state2)

        current_nsamples1 += 1

        # Stopping condition
        if iteration >= max(meeting_time, k):
            finished = True

    samples1 = jnp.stack(samples1[:iteration + 1])
    samples2 = jnp.stack(samples2[:iteration - lag + 1])
    cost = lag + 2*(meeting_time-lag) + iteration-meeting_time

    return {"samples1": samples1, "samples2": samples2, "meetingtime": meeting_time, "iteration": iteration, "finished": finished, "cost": cost,"key": key}



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

    correction_times = jnp.arange(k + lag, meeting_time)
    has_correction = meeting_time > k + lag
    correction_coefficients = (
        jnp.floor_divide(correction_times - k, lag)
        + jnp.floor_divide(
            -jnp.maximum(lag, correction_times - m), lag
        )
        + 1
    )

    estimates = []
    samples1_stop = max(m + 1, meeting_time)
    for h in h_list:
        h_samples1 = jax.vmap(lambda sample: jnp.asarray(h(sample)))(
            samples1[:samples1_stop]
        )
        main_term = jnp.sum(h_samples1[k : m + 1], axis=0)
        correction_term = jnp.zeros_like(jnp.squeeze(h_samples1[0]))

        if has_correction:
            h_samples1_correction = jax.vmap(jnp.squeeze)(
                h_samples1[correction_times]
            )
            h_samples2_correction = jax.vmap(
                lambda sample: jnp.squeeze(h(sample))
            )(samples2[correction_times - lag])
            deltas = h_samples1_correction - h_samples2_correction
            correction_term = jnp.tensordot(
                correction_coefficients, deltas, axes=1
            )

        estimates.append(
            (main_term + correction_term) / (m - k + 1)
        )

    return estimates

# Combine the coupled chain and H_bar to compute the unbiased estimator

def unbiased_estimator(key, kernel, coupled_kernel, initial, h, k, m, lag):
    # Coupled chain
    c_chain = coupled_chain(key, kernel, coupled_kernel, initial, m, lag)

    # Compute H_bar
    H_bar_val = H_bar(c_chain, h, k, m, lag)

    # Add the estimator to the dictionary
    c_chain["uestimator"] = H_bar_val

    return c_chain



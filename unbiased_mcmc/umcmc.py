import numpy as np


def coupled_chain(
	kernel,
	coupled_kernel,
	initial,
	k,
	lag,
	max_iterations=float("inf"),
	preallocate=100,
):
	current_state1, current_pdf1 = initial()
	current_state2, current_pdf2 = initial()
	current_state1 = np.asarray(current_state1)
	current_state2 = np.asarray(current_state2)

	samples1 = [current_state1]
	samples2 = [current_state2]
	iteration = 0

	for _ in range(lag):
		iteration += 1
		current_state1, current_pdf1 = kernel(current_state1, current_pdf1)
		samples1.append(current_state1)

	meet = False
	finished = False
	meeting_time = float("inf")

	while not finished and iteration < max_iterations:
		iteration += 1
		if meet:
			current_state1, current_pdf1 = kernel(current_state1, current_pdf1)
			current_state2 = current_state1
			current_pdf2 = current_pdf1
		else:
			(
				current_state1,
				current_pdf1,
				current_state2,
				current_pdf2,
			) = coupled_kernel(
				current_state1,
				current_pdf1,
				current_state2,
				current_pdf2,
			)

			if np.array_equal(current_state1, current_state2):
				meet = True
				meeting_time = iteration

		samples1.append(current_state1)
		samples2.append(current_state2)

		if iteration >= max(meeting_time, k):
			finished = True

	samples1 = np.stack(samples1[: iteration + 1])
	samples2 = np.stack(samples2[: iteration - lag + 1])
	cost = lag + 2 * (meeting_time - lag) + iteration - meeting_time

	return {
		"samples1": samples1,
		"samples2": samples2,
		"meetingtime": meeting_time,
		"iteration": iteration,
		"finished": finished,
		"cost": cost,
	}


def H_bar(c_chain, h_list, k, m, lag):
	samples1 = c_chain["samples1"]
	samples2 = c_chain["samples2"]
	meeting_time = c_chain["meetingtime"]
	maxiter = c_chain["iteration"]

	if k > maxiter or m > maxiter:
		print("error: k and m must be <= maxiter")
		return None

	correction_times = np.arange(k + lag, meeting_time, dtype=int)
	has_correction = meeting_time > k + lag
	correction_coefficients = (
		np.floor_divide(correction_times - k, lag)
		+ np.floor_divide(-np.maximum(lag, correction_times - m), lag)
		+ 1
	)

	estimates = []
	samples1_stop = max(m + 1, meeting_time)
	for h in h_list:
		h_samples1 = np.stack(
			[np.asarray(h(sample)) for sample in samples1[:samples1_stop]]
		)
		main_term = np.sum(h_samples1[k : m + 1], axis=0)
		correction_term = np.zeros_like(np.squeeze(h_samples1[0]))

		if has_correction:
			h_samples1_correction = np.stack(
				[np.squeeze(value) for value in h_samples1[correction_times]]
			)
			h_samples2_correction = np.stack(
				[
					np.squeeze(h(sample))
					for sample in samples2[correction_times - lag]
				]
			)
			deltas = h_samples1_correction - h_samples2_correction
			correction_term = np.tensordot(
				correction_coefficients, deltas, axes=1
			)

		estimates.append((main_term + correction_term) / (m - k + 1))

	return estimates


def unbiased_estimator(kernel, coupled_kernel, initial, h, k, m, lag):
	c_chain = coupled_chain(kernel, coupled_kernel, initial, m, lag)
	c_chain["uestimator"] = H_bar(c_chain, h, k, m, lag)
	return c_chain

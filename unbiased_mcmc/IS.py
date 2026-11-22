import numpy as np


def IS_estimator(log_weight, proposal, M):
	M = int(M)
	if M < 1:
		raise ValueError("M must be at least 1")

	log_weights = np.asarray(
		[log_weight(proposal()) for _ in range(M)], dtype=float
	)
	return np.logaddexp.reduce(log_weights) - np.log(M)

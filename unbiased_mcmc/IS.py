import numpy as np


def IS_estimator(
	log_weight,
	proposal,
	M,
	batch_log_weight=None,
	batch_proposal=None,
):
	M = int(M)
	if M < 1:
		raise ValueError("M must be at least 1")
	if (batch_log_weight is None) != (batch_proposal is None):
		raise ValueError("batch_log_weight and batch_proposal must be provided together")

	if batch_log_weight is None:
		log_weights = np.asarray(
			[log_weight(proposal()) for _ in range(M)], dtype=float
		)
	else:
		samples = np.asarray(batch_proposal(M))
		if samples.ndim == 0 or samples.shape[0] != M:
			raise ValueError("batch_proposal must return M samples along its first axis")
		log_weights = np.asarray(batch_log_weight(samples), dtype=float)
		if log_weights.shape != (M,):
			raise ValueError("batch_log_weight must return one log weight per sample")

	return np.logaddexp.reduce(log_weights) - np.log(M)

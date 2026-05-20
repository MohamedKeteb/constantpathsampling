import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp

class MixtureMetropolisHastings:
    def __init__(self, X, y, theta_0, Sigma_0, sigma2_noise):
        """
        Metropolis-Hastings sampler for the Zanella defensive mixture q_mix.
        No out-of-sample or analytical LOO predictive knowledge is used.
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.theta_0 = np.asarray(theta_0)
        self.Sigma_0 = np.asarray(Sigma_0)
        self.sigma2 = float(sigma2_noise)
        self.n, self.d = X.shape

    def log_unnormalized_q_mix(self, theta):
        """
        Computes the log of the unnormalized target density \tilde{q}_mix(\theta).
        Formula: log_prior + log_likelihood_full + log( mean( 1 / likelihood_i ) )
        """
        # 1. Log Prior
        log_prior = stats.multivariate_normal.logpdf(theta, mean=self.theta_0, cov=self.Sigma_0)
        
        # 2. Pointwise log-likelihoods for all observations: shape (n,)
        preds = self.X @ theta
        log_lik_vector = stats.norm.logpdf(self.y, loc=preds, scale=np.sqrt(self.sigma2))
        
        # 3. Full Log-Likelihood
        log_lik_full = np.sum(log_lik_vector)
        
        # 4. Log of the average of 1 / p(y_j | \theta)
        # 1 / p(y_j | \theta) = exp(-log_lik_vector)
        # log( mean( exp(-log_lik_vector) ) ) = logsumexp(-log_lik_vector) - log(n)
        log_avg_inv_lik = logsumexp(-log_lik_vector) - np.log(self.n)
        
        # Unnormalized target log-density
        return log_prior + log_lik_full + log_avg_inv_lik

    def sample(self, num_samples=5000, proposal_std=0.1, theta_init=None):
        """
        Runs the Random-Walk Metropolis-Hastings algorithm to sample from q_mix.
        """
        if theta_init is None:
            current_theta = np.zeros(self.d)
        else:
            current_theta = np.array(theta_init)
            
        samples = np.zeros((num_samples, self.d))
        accepted = 0
        
        # Evaluate current state log-density
        current_log_p = self.log_unnormalized_q_mix(current_theta)
        
        for s in range(num_samples):
            # Propose a new state using a Gaussian random walk
            proposal_theta = current_theta + np.random.normal(0, proposal_std, size=self.d)
            
            # Evaluate proposal state log-density
            proposal_log_p = self.log_unnormalized_q_mix(proposal_theta)
            
            # Acceptance probability ratio \alpha = p(prop) / p(curr)
            # In log scale: log(\alpha) = log_p(prop) - log_p(curr)
            log_acceptance_ratio = proposal_log_p - current_log_p
            
            if np.log(np.random.uniform(0, 1)) < log_acceptance_ratio:
                # Accept proposal
                current_theta = proposal_theta
                current_log_p = proposal_log_p
                accepted += 1
                
            samples[s, :] = current_theta
            
        acceptance_rate = accepted / num_samples
        print(f"Metropolis-Hastings complete. Acceptance rate: {acceptance_rate:.2%}")
        return samples


class ZanellaLOOEstimator:
    """
    Computes the out-of-sample LOO score using samples drawn from q_mix.
    """
    def __init__(self, X, y, samples_q_mix, sigma2_noise):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.thetas = np.asarray(samples_q_mix) # Shape: (S, d)
        self.sigma2 = float(sigma2_noise)
        self.S, self.d = self.thetas.shape
        self.n = len(y)

    def estimate_elpd(self):
        # 1. Matrice de log-vraisemblance : Shape (S, n)
        predictions = self.thetas @ self.X.T
        log_lik_matrix = stats.norm.logpdf(self.y, loc=predictions, scale=np.sqrt(self.sigma2))
        
        # 2. Log de q_mix(\theta^s) pour chaque échantillon s : Shape (S,)
        log_q_mix = logsumexp(-log_lik_matrix, axis=1)
        
        loo_log_densities = np.zeros(self.n)
        
        # 3. Calcul par ratio d'importance sampling pour chaque point i
        for i in range(self.n):
            # Log du poids d'importance non-normalisé pour la cible i :
            # log w_i(\theta^s) = log p(\theta^s | y_{-i}) - log q_mix(\theta^s)
            # ce qui équivaut à : -log p(y_i | \theta^s) - log q_mix(\theta^s)
            log_w_s = -log_lik_matrix[:, i] - log_q_mix  # Shape: (S,)
            
            # --- FORMULE : sum( p(y_i | \theta^s) * w_s ) / sum( w_s ) ---
            # En échelle log, le numérateur est : log_sum_exp( log p(y_i | \theta^s) + log w_s )
            log_numerator_sum = logsumexp(log_lik_matrix[:, i] + log_w_s)
            
            # Le dénominateur est : log_sum_exp( log w_s )
            log_denominator_sum = logsumexp(log_w_s)
            
            # Le log de l'estimation finale est la soustraction des deux sums
            loo_log_densities[i] = log_numerator_sum - log_denominator_sum
            
        return loo_log_densities, np.sum(loo_log_densities)

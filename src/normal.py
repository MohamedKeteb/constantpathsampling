
import numpy as np

# Set the random seed for reproducibility and define the parameter D
set_seed(26)
D = 4.0

def log_target0(x):
    return -0.5 * x**2
def log_target1(x):
    return -0.5 * (x - D)**2
def log_target_path(x, path):
    return (1-path) * log_target0(x) + path * log_target1(x)
def grad_log_target_path(x):
    return log_target1(x) - log_target0(x)

def MH_kernel(current_state, current_pdf, path, sigma_proposal):
    proposal = current_state + np.random.normal(0, 1) * sigma_proposal
    proposal_pdf = log_target_path(proposal, path)
    acceptance_prob = min(1, np.exp(proposal_pdf - current_pdf))
    if np.random.rand() < acceptance_prob:
        return proposal, proposal_pdf
    else:
        return current_state, current_pdf

def initial_distribution(path):
    current_state = np.random.normal(loc=-1, scale=2.0)
    current_pdf = log_target_path(current_state, path)
    return current_state, current_pdf

def reflection_maximal_coupling(mu1, mu2, sigma):
    z = (mu1- mu2) / np.sqrt(sigma)
    proposal_x = np.random.normal(0, 1)
    w = np.log(np.random.uniform(0, 1))
    x = mu1 + np.sqrt(sigma) * proposal_x
    if w < -0.5 * ((proposal_x + z)**2) - (-0.5 * proposal_x**2):
        y = x
    else:
        y = mu2 - np.sqrt(sigma) * proposal_x
    return x, y

def MH_coupled_kernel(current_state1, current_pdf1, current_state2, current_pdf2, path, sigma_proposal):
    proposal1, proposal2 = reflection_maximal_coupling(current_state1, current_state2, sigma_proposal)
    proposal_pdf1 = log_target_path(proposal1, path)
    proposal_pdf2 = log_target_path(proposal2, path)
    
    if np.log(np.random.uniform(0, 1)) < proposal_pdf1 - current_pdf1:
        new_state1 = proposal1
        new_pdf1 = proposal_pdf1
    else:
        new_state1 = current_state1
        new_pdf1 = current_pdf1
        
    if np.log(np.random.uniform(0, 1)) < proposal_pdf2 - current_pdf2:
        new_state2 = proposal2
        new_pdf2 = proposal_pdf2
    else:
        new_state2 = current_state2
        new_pdf2 = current_pdf2
        
    return new_state1, new_state2, new_pdf1, new_pdf2








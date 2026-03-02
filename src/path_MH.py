import numpy as np

def MH_kernel(current_state, current_pdf, sigma_proposal, log_target):
    proposal = current_state + np.random.normal(0, 1) * sigma_proposal
    proposal_pdf = log_target(proposal)
    acceptance_prob = min(1, np.exp(proposal_pdf - current_pdf))
    if np.random.rand() < acceptance_prob:
        return proposal, proposal_pdf
    else:
        return current_state, current_pdf

def initial_distribution(log_target, mean_init, sigma_init):
    current_state = np.random.normal(loc=mean_init, scale=sigma_init, size=1)
    current_pdf = log_target(current_state)
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

def MH_coupled_kernel(current_state1, current_pdf1, current_state2, current_pdf2, sigma_proposal, log_target):
    proposal1, proposal2 = reflection_maximal_coupling(current_state1, current_state2, sigma_proposal)
    proposal_pdf1 = log_target(proposal1)
    proposal_pdf2 = log_target(proposal2)
    
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

import numpy as np


def coupled_chain(kernel, coupled_kernel, initial, m, lag, max_iterations = np.inf, preallocate = 10):
    init1 = initial()
    init2 = initial()
    current_state1, current_pdf1 = init1
    current_state2, current_pdf2 = init2
    p = current_state1.shape[0]
    samples1 = np.empty((m + preallocate + lag, p))
    samples2 = np.empty((m + preallocate, p))

    nrowsamples1 = m + preallocate + 1
    
    # Initialisation première ligne
    samples1[0, :] = current_state1
    samples2[0, :] = current_state2
    
    current_nsamples1 = 1   
    iter = 1

    for iter in range(lag):
        current_state1, current_pdf1 = kernel(current_state1, current_pdf1)
        samples1[current_nsamples1, :] = current_state1
        current_nsamples1 += 1


    meet = False
    finished = False
    meeting_time = np.inf
    while not finished and iter < max_iterations:
        


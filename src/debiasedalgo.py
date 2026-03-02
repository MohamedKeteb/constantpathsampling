import numpy as np
import math


def coupled_chain(kernel, coupled_kernel, initial, k, lag, max_iterations = np.inf, preallocate = 10):
    init1 = initial()
    init2 = initial()
    current_state1, current_pdf1 = init1
    current_state2, current_pdf2 = init2
    p = current_state1.shape[0]
    samples1 = np.empty((k + preallocate + lag, p))
    samples2 = np.empty((k + preallocate, p))

    nrowsamples1 = k + preallocate + lag
    
    # Initialisation première ligne
    samples1[0, :] = current_state1
    samples2[0, :] = current_state2
    
    current_nsamples1 = 1   
    iter = 0

    for _ in range(lag):
        iter += 1
        current_state1, current_pdf1 = kernel(current_state1, current_pdf1)
        samples1[current_nsamples1, :] = current_state1
        current_nsamples1 += 1


    meet = False
    finished = False
    meeting_time = np.inf
    while not finished and iter < max_iterations:
        iter += 1
        if meet:
            current_state1, current_pdf1 = kernel(current_state1, current_pdf1)
            current_state2, current_pdf2 = current_state1, current_pdf1
        else:
            current_state1, current_state2, current_pdf1, current_pdf2 = coupled_kernel(current_state1, current_pdf1, current_state2, current_pdf2)
            if not meet and np.array_equal(current_state1, current_state2):
                meet = True
                meeting_time = iter
        if current_nsamples1 >= nrowsamples1:
            
            samples1 = np.vstack((samples1,np.full((nrowsamples1, p), np.nan)))
            samples2 = np.vstack((samples2,np.full((nrowsamples1, p), np.nan)))
            nrowsamples1 = 2*nrowsamples1

        samples1[current_nsamples1, :] = current_state1
        samples2[current_nsamples1-lag, :] = current_state2
        current_nsamples1 += 1

        if iter >= max(meeting_time, k):
            finished = True

        return {"samples1": samples1, "samples2": samples2, "meetingtime": meeting_time, "iteration": iter, "finished": finished}
    


def H_bar(c_chain, h_list, k, m, lag):

    samples1 = c_chain["samples1"]
    samples2 = c_chain["samples2"]
    meeting_time = c_chain["meetingtime"]
    maxiter = c_chain["iteration"]

    # -------- Checks --------
    if k > maxiter:
        print("error: k must be <= maxiter")
        return None

    if m > maxiter:
        print("error: m must be <= maxiter")
        return None
    
    # -------- Terme principal --------

    H_bar_val = np.zeros(len(h_list))

    for i, h in enumerate(h_list):
        h_of_chain = np.array([h(x) for x in samples1[k:m+1, :]])
        
        # ensure 2D
        h_of_chain = np.atleast_2d(h_of_chain)

        H_bar_val[i] = np.sum(h_of_chain, axis=0)

    # -------- Terme de correction --------
    deltas_term = np.zeros(H_bar_val.shape)

    for i, h in enumerate(h_list):
        if meeting_time > k + lag:

            for t in range(k+lag, meeting_time):

                coefficient = math.floor((t - k) / lag) - math.ceil(max(lag, t - m) / lag) + 1
                
                delta = (
                    np.atleast_1d(h(samples1[t, :])) -
                    np.atleast_1d(h(samples2[t - lag, :]))
                )

                deltas_term[i] += coefficient * delta
        else: pass

    return (H_bar_val + deltas_term) / (m- k + 1)


def unbiased_estimator(kernel, coupled_kernel, initial, h, k, m, lag):
    # --- Coupled chain ---
    c_chain = coupled_chain(kernel, coupled_kernel, initial, m, lag)

    # --- computation of H_bar ---
    H_bar_val = H_bar(c_chain, h, k, m, lag)

    # --- add  H_bar_val to the dictionary ---
    c_chain["H_bar_val"] = H_bar_val

    return c_chain






    



    

    


        





        
    


        


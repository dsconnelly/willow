import numpy as np

def R_squared(Y, output):
    means = Y.mean(axis=0)
    ss_res = ((Y - output) ** 2).sum(axis=0)
    ss_tot = ((Y - means) ** 2).sum(axis=0)
    
    mask = ss_tot != 0
    output = np.zeros(Y.shape[1])
    output[mask] = 1 - (ss_res[mask] / ss_tot[mask])
    
    return output
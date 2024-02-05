import numpy as np
import h5py
from tqdm import tqdm
import config as c
import pickle
import os

from static_emc_init import *

import utils_cl
import utils

# load recon data 
a = pickle.load(open('recon.pickle', 'rb'))

# i -> pixel index
# d -> frame index
# t -> class index
# l -> background class index

# fluence w[d]
w = a.w

# classes W[t, i]
W = a.W

# background weights for each frame b[d, l]
b = a.b

# background classes B[l, i]
B = a.B

# log likelihood for each frame and class LR[d, c]
LR = a.LR

# probability matrix P[d, c]
P = a.P

# transpose of above  PT[c, d]
#PT = a.PT

# photon counts K[d, i]
K = a.K
inds = a.inds

# transpose of above KT[i, d]
#KT = a.KT
#indsT = a.indsT

print('classes:', W.shape[0])
print('frames:', len(K))
print('pixels:', W.shape[1])

beta = 0.002

print('iterations:', a.iterations)

#for i in range(c.iters):
for i in range(1):
    # this effectively sets the baseline background scale to 1
    # since b is intitialised to 1
    #if a.iterations == 0 :
    if i == 0 :
        update_b = False
    else :
        update_b = True
    
    LL, E = utils_cl.calculate_P(K, inds, w, W, b, B, LR, P, beta)
    
    #utils_cl.update_w(P, w, W, b, B, K, inds, tol_P = 1e-3, tol = 1e-5, min_val = 1e-3, update_b = update_b)
    
    #utils_cl.update_W(P, w, W, K, inds, b, B, tol_P = 1e-3, update_B = True)
    
    # keep track of log-likelihood values
    a.most_likely_classes.append(np.argmax(P, axis=1))
    a.LL.append(LL)
    a.expectation_values.append(E)
    a.iterations += 1
    #utils.plot_iter(a, a.iterations)
    #os.system("pdfunite recon_*.pdf recon.pdf")
    
    # save state
    pickle.dump(a, open('recon_cpu.pickle', 'wb'))

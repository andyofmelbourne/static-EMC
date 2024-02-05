import numpy as np

PREFIX = '/home/andyofmelbourne/Documents/2023/P3004-take-2/gold/'
mask   = PREFIX + 'badpixel_mask_r0096.h5'
data   = PREFIX + 'hits_r0087.cxi'
dataT  = PREFIX + 'hits_r0087_T.cxi'

classes            = 200
background_classes = 1
max_frames         = 5000
frame_shape        = (16, 128, 512)
# just use the first part of the first panel (low q)
pixels             = np.arange(1024**2).reshape(frame_shape)[0, :, :128]


tol_P = 1e-2

#iters = 20
#update_b = np.zeros((iters,), dtype=bool)
#update_B = np.zeros((iters,), dtype=bool)
#beta_start = 0.001
#beta_stop  = 0.1
#betas = (beta_stop / beta_start)**(np.arange(iters)/(iters-1)) * beta_start

#iters = 5
#update_b = np.zeros((iters,), dtype=bool)
#update_B = np.zeros((iters,), dtype=bool)
#betas = 0.1 * np.ones((iters,))

#iters = 3
#update_b = np.zeros((iters,), dtype=bool)
#update_b[1:] = True
#update_B = np.ones((iters,), dtype=bool)
#betas = 0.1 * np.ones((iters,))

iters = 5
update_b = np.zeros((iters,), dtype=bool)
update_b[1:] = True
update_B = np.ones((iters,), dtype=bool)
betas = 0.1 * np.ones((iters,))

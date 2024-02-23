import numpy as np

PREFIX = '/gpfs/exfel/exp/SQS/202302/p003004/scratch/'

data = []
for i in range(87, 96):
    data.append(PREFIX + f'saved_hits/hits_r00{i}.cxi')

classes            = 1000
background_classes = 1
max_frames         = 1000000
frame_shape        = (16, 128, 512)
imshow             = lambda x: x[0, :, :128]
# just use the first part of the first panel (low q)
pixels             = imshow(np.arange(1024**2).reshape(frame_shape))
filter_by          = None
filter_value       = None

tol_P = 1e-2

iters = 40
update_b     = np.ones((iters,), dtype=bool)
update_b[0]  = False
update_B     = np.zeros((iters,), dtype=bool)
beta_start   = 0.001
beta_stop    = 0.1
betas = (beta_stop / beta_start)**(np.arange(iters)/(20-1)) * beta_start

betas[20:] = beta_stop

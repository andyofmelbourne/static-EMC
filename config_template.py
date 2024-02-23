import numpy as np

PREFIX = '/home/andyofmelbourne/Documents/2023/P3004-take-2/gold/'

data = []
for i in range(87, 96):
    data.append(PREFIX + f'hits_r00{i}.cxi')

classes            = 200
background_classes = 1
max_frames         = 5000
frame_shape        = (16, 128, 512)
imshow             = lambda x: x[0, :, :128]
# just use the first part of the first panel (low q)
pixels             = imshow(np.arange(1024**2).reshape(frame_shape))
filter_by          = 'static_emc/good_hit' # or None

tol_P = 1e-2

iters = 20
update_b = np.zeros((iters,), dtype=bool)
update_B = np.zeros((iters,), dtype=bool)
beta_start = 0.001
beta_stop  = 0.1
betas = (beta_stop / beta_start)**(np.arange(iters)/(iters-1)) * beta_start

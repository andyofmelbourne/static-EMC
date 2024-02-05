import numpy as np

PREFIX = '/home/andyofmelbourne/Documents/2023/P3004-take-2/gold/'
mask   = PREFIX + 'badpixel_mask_r0096.h5'
data   = PREFIX + 'hits_r0087.cxi'
dataT  = PREFIX + 'hits_r0087_T.cxi'

classes            = 100
background_classes = 1
max_frames         = 1000
frame_shape        = (16, 128, 512)
# just use the first part of the first panel (low q)
pixels             = np.arange(1024**2).reshape(frame_shape)[0, :, :128]

beta  = 0.1
iters = 1



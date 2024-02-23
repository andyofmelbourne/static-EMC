import pickle
import numpy as np
import utils
import h5py
import sys
import scipy.special
from scipy.stats import poisson 
from tqdm import tqdm

from static_emc_init import A

# load good class list from reconstruction file
# need cxi files to get good classes
# which means I need config.py
# need recon file to get variables
# then label frames with surprisingly low Log-likelihood values as bad

gui_check = False

if gui_check :
    import matplotlib.pyplot as plt
    import pyqtgraph as pg

print('loading configuration file:', sys.argv[1])
config = utils.load_config(sys.argv[1])

with h5py.File(config.data[0]) as f:
    good_classes = f['static_emc/good_classes'][()]

# load photons
print('loading photons:', sys.argv[2])
K, inds = pickle.load(open(sys.argv[2], 'rb'))

# load recon file
print('loading reconstruction file:', sys.argv[3])
r = pickle.load(open(sys.argv[2], 'rb'))

I  = r.I
T  = np.empty((I,), dtype = float)
k  = np.empty((I,), dtype = int)

# speed things up by precalculating entropy
#wmax = r.w.max() * r.W.max()
#lam = np.linspace(0, wmax, 1000)
#entropy = poisson.entropy(lam)
# entropy doesn't seem to be giving "typical" logpmf values...

bad_frames_cxi = {}
for fnam in config.data :
    bad_frames_cxi[fnam] = []

number_of_frames_per_class     = np.empty((len(good_classes)), dtype=int)
number_of_bad_frames_per_class = np.empty((len(good_classes)), dtype=int)

for class_index, good_class in tqdm(enumerate(good_classes), total=len(good_classes), desc = 'finding bad frames in classes') :
    ds = np.where(r.most_likely_classes[-1] == good_class)[0]
    #print('found', len(ds),'frames with class', good_class,' as most likely class')
    
    D  = len(ds)
    number_of_frames_per_class[class_index] = D
    
    ksums        = np.empty((D,), dtype = int)
    LLs          = np.empty((D,), dtype = float)
    expected_LLs = np.empty((D,), dtype = float)

    for i, d in tqdm(enumerate(ds), total = D, desc = 'calculating LLs', leave=False) :
        # photons per pattern
        ksums[i] = np.sum(K[d])
    
        # calculate likelihoods
        k.fill(0)
        k[inds[d]] = K[d]
        T[:]       = r.w[d] * r.W[good_class] + np.dot(r.b[d], r.B)
        LLs[i]     = np.sum(poisson.logpmf(k, T)) #np.sum( k * np.log(T) - T - scipy.special.gammaln(1+k))
        
        # calculate expected likelihoods
        #expected_LLs[i] = -np.sum(poisson.entropy(T))
        #expected_LLs[i] = -np.sum(np.interp(T, lam, entropy))
        expected_LLs[i] = np.sum(poisson.logpmf(poisson.rvs(T), T))
        
        
    
    # now fit quadratic to expected_LLs vs log(ksums) for thresholding
    m2, m1, c = np.polyfit(np.log(ksums), expected_LLs, 2)
    lk        = np.log(ksums)
    y         = m2 * lk**2 + m1 * lk + c
    LL_min    = 1.1 * y
    
    bad_frames = LLs < LL_min
    
    number_of_bad_frames_per_class[class_index] = np.sum(bad_frames)
    #print('found', np.sum(bad_frames), 'bad frames')

    for i, d in enumerate(np.where(bad_frames)[0]):
        bad_frames_cxi[config.data[r.file_index[d]]].append(r.frame_index[d])
    
    if gui_check :
        # make 2d frames
        image   = np.empty(r.frame_shape, dtype=float)
        imshape = image[r.frame_slice].shape
        frames  = np.zeros((len(ds),) + imshape, dtype=np.float32)
        
        for i, d in enumerate(ds):
            k.fill(0)
            image.fill(0)
            
            k[inds[d]] = K[d]
            
            image.ravel()[r.pixel_indices] = k
            frames[i] = image[r.frame_slice] 

        fig, ax = plt.subplots()
        color = ['r' if bad_frames[d] else 'b' for d in range(len(bad_frames))]
        ax.scatter(ksums, LLs, picker = 2, c = color, s = 2, label = 'log-likelihood for good (blue) and bad (red) frames')
        ax.scatter(ksums, expected_LLs, c = 'k', s = 2, label = 'typical log-likelihood')
        #ax.scatter(ksums, y, c = 'g', s = 2, label = 'tolerable range')
        
        x  = np.linspace(ksums.min(), ksums.max(), 1000)
        lx = np.log(x)
        y2 = m2 * lx**2 + m1 * lx + c
        ax.fill_between(x, 1.1 * y2, 0, alpha = 0.3, label='tolerable range')
        ax.set_ylabel('Log-likelihood')
        ax.set_xlabel('photons in frame')
        ax.legend()

        def on_pick(event):
            i = event.ind[0]
            print('frame:', i, 'LL:', round(LLs[i]), 'expected LL:', round(expected_LLs[i]), 'LL-measure:', round(LLs[i] - 1.1 * y[i]))
            pg.show(frames[event.ind])

        fig.canvas.mpl_connect('pick_event', on_pick)
        plt.show()

print('{:<10} {:<30} {:<30}'.format('class', 'number of frames', 'number of bad frames'))
for class_index, good_class in enumerate(good_classes):
    print('{:<10} {:<30} {:<30}'.format( good_class, number_of_frames_per_class[class_index], number_of_bad_frames_per_class[class_index] ))

# hack to recover from 

# write to cxi files
for fnam in tqdm(config.data, 'writing bad frames to cxi files') :
    ds = np.array(bad_frames_cxi[fnam])
    if len(ds) != 0 :
        with h5py.File(fnam, 'a') as f:
            gh = f['static_emc/good_hit'][()]
            gh[ds] = False
            f['static_emc/good_hit'][:] = gh

    

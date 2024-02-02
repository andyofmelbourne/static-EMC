import numpy as np
import h5py
from tqdm import tqdm
import pickle

#np.random.seed(1)

class A():
    def __init__(self, C, L, D, I, K, inds, mask, pixel_indices, beta):
        self.beta = beta
        self.C = C
        self.L = L
        self.D = D
        self.I = I
        self.K = K
        self.inds = inds
        #self.KT = KT
        #self.indsT = indsT
        self.mask = mask
        
        self.LL = []
        self.expectation_values = []
        self.iterations = 0
         
        self.pixel_indices = pixel_indices
        
        self.LR = np.empty((D, C), dtype = np.float32)
        self.P  = np.zeros((D, C), dtype = np.float32)
        #self.PT = np.empty((C, D), dtype = np.float32)
        
        self.w = np.ones((D,), dtype = np.float32)
        self.b = np.ones((D, L), dtype = np.float32)
        self.W = np.ascontiguousarray(np.random.random((C, I)).astype(np.float32))
        self.B = 1e-3 * np.ascontiguousarray(np.random.random((L, I)).astype(np.float32))
        #self.B = np.ascontiguousarray(np.random.random((L, I)).astype(np.float32))

def init(c):
    output = 'recon.pickle'

    # load mask    
    print('loading mask:', c.mask)
    with h5py.File(c.mask) as f:
        mask0 = f['entry_1/good_pixels'][()]
        frame_shape = mask0.shape
        frame_size  = mask0.size
    
    # now set all pixels not selected in config to False
    mask = np.zeros_like(mask0)
    mask.ravel()[c.pixels] = mask0.ravel()[c.pixels]
    
    # store the un-masked pixel indices (flattened) 
    # which will be used for the reconstruction
    pixel_indices = np.arange(frame_size)[mask.ravel()]
    
    I = np.sum(mask)
    
    # load data into sparse array
    K    = []
    inds = []
    inds_f = np.arange(I, dtype = np.int64)
    print('loading data:', c.mask)
    with h5py.File(c.data) as f:
        data = f['entry_1/data_1/data']
        D    = min(data.shape[0], c.max_frames)
        
        for d in tqdm(range(D)):
            frame = data[d][mask].ravel()
            m = frame > 0 
            K.append(frame[m].copy())
            inds.append(inds_f[m].copy())
        
    """
    # load transposed data into sparse array
    KT    = []
    indsT = []
    
    # frame indices
    inds_d = np.arange(D, dtype=np.int64)
    
    # un-masked and selected pixel indices
    inds_i = np.arange(mask.size, dtype = np.int64).reshape(mask.shape)[mask]
    
    print('loading transposed data:', c.mask)
    with h5py.File(c.dataT) as f:
        data = f['data_id']
        
        for i in tqdm(pixel_indices):
            frame = data[i, :D]
            m = frame > 0 
            KT.append(frame[m])
            indsT.append(inds_d[m])
    """
    
    print(f'Found {D} frames with {I} unmasked pixels')
            
    a = A(c.classes, c.background_classes, D, I, K, inds, mask, pixel_indices, c.beta)
    
    # save sparse datasets        
    print('saving reconstruction variables to:', output)
    pickle.dump(a, open(output, 'wb'))

if __name__ == '__main__' :
    import config
    init(config)

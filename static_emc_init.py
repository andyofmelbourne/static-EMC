import numpy as np
import h5py
from tqdm import tqdm
import pickle

np.random.seed(1)

class A():
    def __init__(self, C, L, D, I, K, inds, mask, B, pixel_indices, file_index, frame_index, beta):
        self.betas = beta
        self.C = C
        self.L = L
        self.D = D
        self.I = I
        self.K = K
        self.inds = inds
        #self.KT = KT
        #self.indsT = indsT
        self.mask = mask
        
        self.most_likely_classes = []
        self.LL = []
        self.expectation_values = []
        self.iterations = 0
         
        self.pixel_indices = pixel_indices
        self.file_index = file_index
        self.frame_index = frame_index
        
        self.LR = np.empty((D, C), dtype = np.float32)
        self.P  = np.zeros((D, C), dtype = np.float32)
        #self.PT = np.empty((C, D), dtype = np.float32)
        
        self.w = np.ones((D,), dtype = np.float32)
        self.b = np.ones((D, L), dtype = np.float32)
        self.W = 1e-3 + np.ascontiguousarray(np.random.random((C, I)).astype(np.float32))
        self.B = np.zeros((L, I), dtype = np.float32)
        
        if type(B) is not type(None) :
            self.B[0] = B

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
    file_index = []
    frame_index = []
    inds_f = np.arange(I, dtype = np.int64)
    print('loading data:', c.data)
    if type(c.data) is str :
        fnams = [c.data]
    else :
        fnams = c.data
    
    for i, fnam in enumerate(fnams):
        with h5py.File(fnam) as f:
            data = f['entry_1/data_1/data']
            D    = min(data.shape[0], c.max_frames)
            
            for d in tqdm(range(D)):
                frame = data[d][mask].ravel()
                m = frame > 0 
                if np.sum(frame[m]) > 10 :
                    K.append(frame[m].copy())
                    inds.append(inds_f[m].copy())
                    file_index.append(i)
                    frame_index.append(d)
    
    # load background 
    if c.background is not None :
        print('loading background:', c.data)
        with h5py.File(c.background) as f:
            B = f['data'][()][mask].ravel()
    else :
        B = None
        
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
    
    D = len(K)
    print(f'Found {D} frames with {I} unmasked pixels')
            
    a = A(c.classes, c.background_classes, D, I, K, inds, mask, B, pixel_indices, file_index, frame_index, c.betas[0])
    
    # save sparse datasets        
    print('saving reconstruction variables to:', output)
    pickle.dump(a, open(output, 'wb'))

if __name__ == '__main__' :
    import config
    init(config)

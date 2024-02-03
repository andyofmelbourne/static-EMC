import numpy as np
import pyopencl as cl
import pyopencl.array 
from tqdm import tqdm

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

gpu_precision = np.float32

# find an opencl device (preferably a GPU) in one of the available platforms
for p in cl.get_platforms():
    devices = p.get_devices(cl.device_type.GPU)
    if len(devices) > 0:
        break
    
if len(devices) == 0 :
    for p in cl.get_platforms():
        devices = p.get_devices()
        if len(devices) > 0:
            break

context = cl.Context(devices)
queue   = cl.CommandQueue(context)

cl_code = cl.Program(context, open('utils.c', 'r').read()).build()


# find an opencl device (preferably a CPU) in one of the available platforms
for p in cl.get_platforms():
    devices = p.get_devices(cl.device_type.CPU)
    if len(devices) > 0:
        break

context_cpu = cl.Context(devices)
queue_cpu   = cl.CommandQueue(context_cpu)

cl_code_cpu = cl.Program(context_cpu, open('utils.c', 'r').read()).build()

# make an iterator that splits N into chunks of size n
class chunk_csize:
    def __init__(self, N, n):
        self.chunks = int(np.ceil(N/n))
        self.istart = np.int32(-n)
        self.n      = n
        self.N      = N
        self.counter = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        self.counter += 1
        self.istart  += self.n
        self.istop   = np.int32(min(self.istart + self.n, self.N))
        if self.counter <= self.chunks :
            return (self.istart, self.istop, np.int32(self.istop - self.istart))
        raise StopIteration

    def __len__(self):
        return self.chunks

        

def calculate_LR(K, inds, w, W, b, B, LR, beta, min_val = 1e-10):
    """
    each rank processes all frames for a given set of classes
    LR[d, t] = sum_i K[d, i] log( T[t, d, i] ) - T[t, d, i]
    tranpose for faster summing
    LR[d, t] = sum_i K[i, d] log( T[i, t, d] ) - T[i, t, d]
    """
    D = np.int32(w.shape[0])
    C = np.int32(W.shape[0])
    L = np.int32(b.shape[1])
    I = np.int32(W.shape[1])
    
    beta = np.float32(beta)
    print('\nProbability matrix calculation')
    
    # split classes by rank
    my_classes = list(range(rank, C, size))
    classes    = np.int32(len(my_classes))
    
    # parallelise over d chunks on gpu
    frames = np.int32(1024)
    
    LR_cl = cl.array.zeros(queue, (frames,), dtype = np.float32)
    w_cl  = cl.array.empty(queue, (frames,), dtype = np.float32)
    W_cl  = cl.array.empty(queue, (I,),      dtype = np.float32)
    b_cl  = cl.array.empty(queue, (frames, L),       dtype = np.float32)
    B_cl  = cl.array.empty(queue, (L, I),            dtype = np.float32)
    K_cl  = cl.array.empty(queue, (I, frames),       dtype = np.uint8)
    
    K_dense = np.zeros((I, frames), dtype = np.uint8)

    LR_buf = np.empty((frames,), dtype = np.float32)
    
    if rank == 0 :
        disable = False
    else :
        disable = True

    # loop over classes
    for c in tqdm(my_classes, desc='processing class', disable = disable):
        cl.enqueue_copy(queue, W_cl.data, W[c])
        cl.enqueue_copy(queue, B_cl.data, B)
            
        for dstart, dstop, dd in tqdm(chunk_csize(D, frames), desc = 'processing frames', leave = False, disable = disable):
            # load transposed photons to gpu 
            K_dense.fill(0)
            for d in range(dstart, dstop):
                K_dense[inds[d], d] = K[d]
            cl.enqueue_copy(queue, K_cl.data, K_dense)
            
            # load class etc. to gpu
            cl.enqueue_copy(queue, w_cl.data, w[dstart:dstop])
            cl.enqueue_copy(queue, b_cl.data, b[dstart:dstop])
            
            cl_code.calculate_LR_T_dt(queue, (dd,), None, 
                    LR_cl.data,  K_cl.data, w_cl.data, W_cl.data,
                    b_cl.data, B_cl.data, beta, L, I, frames)
            
            cl.enqueue_copy(queue, LR_buf, LR_cl.data)
            LR[dstart: dstop, c] = LR_buf[:dd]
    
def calculate_expectation(K, inds, w, W, b, B, LR, P, beta):
    calculate_LR(K, inds, w, W, b, B, LR, beta)
    
    # E = sum_dt P[d, t] LR[d, t]
    expectation = np.sum(P * LR)
    return expectation

def calculate_P(K, inds, w, W, b, B, LR, P, beta):
    LR.fill(0)
    calculate_LR(K, inds, w, W, b, B, LR, beta)

    x = np.zeros_like(LR)
    comm.Allreduce(LR, x, op = MPI.SUM)
    LR[:] = x
    
    # calculate log-likelihood before normalisation
    LL = np.sum(LR)
    
    # E = sum_dt P[d, t] LR[d, t]
    expectation = np.sum(P * LR) / beta
    
    normalise_LR(LR, P)
    return LL, expectation
    
def normalise_LR(LR, P):
    """
    P[d,t] = exp(LR[d,t]) / sum_t exp(LR[d,t])
    """
    # normalise to produce probability matrix
    m = np.max(LR, axis=1)
    
    LR[:] = np.exp(LR - m[:, None])
    P[:]  = LR / np.sum(LR, axis=-1)[:, None]
    
def update_W(P, w, W, K, inds, b, B, tol_P = 1e-2, minval = 1e-10, update_B = True):
    """
    use dense K so we can parallelise
    but still mask based on P[t, d]
    
    grad[t, i] = sum_d P[d, t] w[d] ( K[i, d] / T[t, d, i] - 1)
               = sum_d P[d, t] w[d] K[i, d] / (w[d] W[t, i] + B[d, i]) - sum_d P[d, t] w[d] 
               = sum_d P[d, t] K[i, d] / (W[t, i] + B[d, i] / w[d]) - g0[t]
    
    - fast (sum over slow axis)
    g0[t] = sum_d P[d, t] w[d]
    
    - fast (d mask + sum over slow axis)
    - but we will only be able to store K[d, i] when P is sparse
    - then we will have to chunk over d 
    - one worker per t i index
    xmax[t, i] = sum_d P[d, t] K[d, i] / g0[t]
    
    - fast
    f[t, i] =  sum_d P[d, t] K[d, i] / (W[t, i] + B[d, i] / w[d]) 
    g[t, i] = -sum_d P[d, t] K[d, i] / (W[t, i] + B[d, i] / w[d])^2 
    
    - do not want to store on gpu
    B[d, i] = b[d, l] B[l, i]
    """
    # check if things will fit in memory
    C = np.int32(W.shape[0])
    I = np.int32(W.shape[1])
    D = np.int32(w.shape[0])
    L = np.int32(B.shape[0])
        
    minval = np.float32(minval)
    
    max_bytes = 10 * 1024**3 
    bytes = 4 * C * I * 4 + D * I * 1
    # needed pixel chunks
    chunks = np.ceil(bytes / max_bytes)
   
    print('\nW-update')
    print('chunks:', chunks)
    
    pixels = np.int32(256) 
    
    # calculate g0
    P_cl = cl.array.empty(queue, (D * C,), dtype = np.float32)
    w_cl  = cl.array.empty(queue, (D,), dtype = np.float32)
    K_cl  = cl.array.empty(queue, (D * pixels,), dtype = np.uint8)
    ds_cl = cl.array.empty(queue, (C * D), dtype = np.int32)
    Ds_cl = cl.array.empty(queue, (C,), dtype = np.int32)
    W_cl  = cl.array.empty(queue, (C * pixels), dtype = np.float32)
    B_cl  = cl.array.empty(queue, (L * pixels), dtype = np.float32)
    b_cl  = cl.array.empty(queue, (L * D), dtype = np.float32)

    gW_cl  = cl.array.empty(queue, (C,), dtype = np.float32)
    gB_cl  = cl.array.empty(queue, (L,), dtype = np.float32)
    background_cl  = cl.array.empty(queue, (D * pixels), dtype = np.float32)

    K_dense = np.zeros((D, pixels,), dtype = np.uint8)

    cl.enqueue_copy(queue, P_cl.data, P)
    cl.enqueue_copy(queue, w_cl.data, w)
    cl.enqueue_copy(queue, b_cl.data, b)

    W_buf = np.empty((C, pixels), dtype=W.dtype)
    B_buf = np.empty((L, pixels), dtype=B.dtype)
    
    ds = np.zeros((C, D), dtype = np.int32)
    Ds = np.zeros((C,), dtype = np.int32)
    for i in tqdm(range(1), desc='masking low P-value frames'):
        # generate the frame lookup table 
        for t in range(C):
            # good frame P[d, t] > Pmax[t]
            p = P[:, t] 
            d = np.where(p > (p.max() * tol_P))[0]
            ds[t, :d.shape[0]] = d
            Ds[t] = d.shape[0]
        
        cl.enqueue_copy(queue, ds_cl.data, ds)
        cl.enqueue_copy(queue, Ds_cl.data, Ds)
    
    for i in tqdm(range(1), desc='calculating gw[t]'):
        cl_code.calculate_gW(queue, (C,), None,
            P_cl.data, w_cl.data, ds_cl.data, Ds_cl.data, 
            gW_cl.data, C, D)

    for t in tqdm(range(1), desc='calculating gB[l]'):
        cl_code.calculate_gB(queue, (L,), None,
            b_cl.data, gB_cl.data, L, D)
    
    for istart, istop, di in tqdm(chunk_csize(I, pixels), desc='updating W & B'):
        # load photons to gpu 
        K_dense.fill(0)
        for d in range(D):
            m = (inds[d] >= istart) * (inds[d] < istop)
            K_dense[d, inds[d][m] - istart] = K[d][m]
        cl.enqueue_copy(queue, K_cl.data, K_dense)
         
        W_buf[:, :di] = W[:, istart:istop]
        B_buf[:, :di] = B[:, istart:istop]
        cl.enqueue_copy(queue, W_cl.data, W_buf)
        cl.enqueue_copy(queue, B_cl.data, B_buf)
        
        cl_code.calculate_background(queue, (D, di), None,
                B_cl.data, b_cl.data, w_cl.data, background_cl.data, pixels, L, D)
    
        cl_code.update_W(queue, (C, di), None, 
                         P_cl.data, K_cl.data, 
                         ds_cl.data, Ds_cl.data, gW_cl.data, background_cl.data, 
                         W_cl.data, w_cl.data, minval, pixels, C, D)
        
        cl.enqueue_copy(queue, W_buf, W_cl.data)
        W[:, istart:istop] = W_buf[:, :di]

        # now that P and K are on the gpu may as well update background
        if update_B :
            cl_code.update_B(queue, (L, di), None, 
                             P_cl.data, K_cl.data, 
                             ds_cl.data, Ds_cl.data, gB_cl.data, B_cl.data, b_cl.data,
                             W_cl.data, w_cl.data, minval, L, pixels, C, D)
            
            cl.enqueue_copy(queue, B_buf, B_cl.data)
            B[:, istart:istop] = B_buf[:, :di]
    
    W[:] = np.clip(W, minval, None)
    B[:] = np.clip(B, minval, None)
    
    """
    grad[l, i] = sum_dt P[d, t] b[d, l] (K[d, i] / T[l, d, t, i] - 1)
               = sum_dt P[d, t] b[d, l] K[d, i] / T[l, d, t, i] - sum_dt P[d, t] b[d, l]
               = sum_dt P[d, t] b[d, l] K[d, i] / (w[d] W[t, i] + sum_l b[d, l] B[l, i]) - sum_d b[d, l]
               = sum_dt P[d, t] b[d, l] K[d, i] / (b[d] B[i] + w[d] W[t, i] + sum_l'neql b[d, l'] B[l', i]) - g0[l]
               = sum_dt P[d, t] K[d, i] / ( B[i] + (w[d] W[t, i] + sum_l'neql b[d, l'] B[l', i]) / b[d]) - g0[l]

    g0[l] = sum_t (sum_d P[t, d] b[d, l])
    """

def update_w(P, w, W, b, B, K, inds, tol_P = 1e-3, tol = 1e-5, min_val = 1e-10, max_iters=1000, update_b = True):
    """
    gw[d] = sum_t P[d, t] sum_i W[t, i] (K[d, i] / T[i, t, d] - 1)
          = sum_t P[d, t] sum_i W[t, i] K[d, i] / T[i, t, d] - sum_t P[d, t] sum_i W[t, i]
          = sum_t P[d, t] sum_i W[t, i] K[d, i] / (w[d] W[t, i] + sum_l b[d, l] B[l, i]) - sum_t P[d, t] sum_i W[t, i]
          = sum_t P[d, t] sum_i K[d, i] / (w[d] + sum_l b[d, l] B[l, i] / W[t, i]) - g0[d]
    
    
    gb[d, l] = sum_t P[d, t] sum_i B[l, i] (K[d, i] / T[i, t, d, l] - 1)
             = sum_t P[d, t] sum_i B[l, i] K[d, i] / (w[d] W[t, i] + sum_l' b[d, l'] B[l', i]) - (sum_t P[d, t]) (sum_i B[l, i])
             = sum_t P[d, t] sum_i K[d, i] / (b[d, l] + (w[d] W[t, i] + sum_l'!=l b[d, l'] B[l', i])/B[l, i]) - sum_i B[l, i]
    """
    # check if things will fit in memory
    C = np.int32(W.shape[0])
    I = np.int32(W.shape[1])
    D = np.int32(w.shape[0])
    L = np.int32(B.shape[0])

    # each rank processes a sub-set of frames
    my_frames = list(range(rank, C, size))
    
    if rank == 0 :
        print('\nupdating fluence estimates')
        disable = False
    else :
        disable = True

    Wsums = np.sum(W[my_frames, :], axis=-1)
    g0    = np.dot(Wsums, P[my_frames, :])
    Ksums = np.array([np.sum(K[d]) for d in my_frames])
    xmax  = Ksums / g0
    
    for d in tqdm(my_frames, desc = 'processing frame', disable = disable):
        p = P[d] 
        classes    = np.where(p > (p.max() * tol_P))[0]
        background = np.dot(b[d], B)
        t, i = np.ix_(classes, inds[d])
        Wti  = np.ascontiguousarray(W[t, i])
        Bi   = np.ascontiguousarray(background[i])
        Pt   = np.ascontiguousarray(P[d][t])
        PK   = np.ascontiguousarray(Pt * K[d])
        x    = np.clip(w[d], min_val, xmax[d])
        c    = g0[d]
    
        
        for iter in range(3):
            T = x + Bi / Wti
            f =  np.sum(PK / T)
            g = -np.sum(PK / T**2)
             
            u  = - f * f / g 
            v  = - f / g - x 
            xp = u / c - v 
             
            x = np.clip(xp, min_val, xmax[d]) 
        
        w[d] = x;
    
    # all gather
    w_all = np.empty_like(w)
    for r in range(rank):
        rank_frames = list(range(rank, C, size))
        w_all[rank_frames] = comm.bcast(w[rank_frames], root=rank)
    w[:] = w_all



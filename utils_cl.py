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
        
    minval = np.float32(min_val)
    
    # chunk over data
    frames = np.int32(2048)
    
    P_cl = cl.array.empty(queue, (frames * C,), dtype = np.float32)
    w_cl  = cl.array.empty(queue, (frames,), dtype = np.float32)
    K_cl  = cl.array.empty(queue, (frames * I,), dtype = np.uint8)
    ds_cl = cl.array.empty(queue, (C, frames,), dtype = np.int32)
    Ds_cl = cl.array.empty(queue, (C,), dtype = np.int32)
    W_cl  = cl.array.empty(queue, (C * I,), dtype = np.float32)
    B_cl  = cl.array.empty(queue, (L * I,), dtype = np.float32)
    b_cl  = cl.array.empty(queue, (L * frames,), dtype = np.float32)
    
    ts_cl         = cl.array.empty(queue, (frames * C,), dtype = np.int32)
    Ts_cl         = cl.array.empty(queue, (frames,),     dtype = np.int32)
    Wsums_cl      = cl.array.empty(queue, (C,), dtype = np.float32)
    gw_cl         = cl.array.empty(queue, (frames,), dtype = np.float32)
    gb_cl         = cl.array.empty(queue, (L,), dtype = np.float32)
    background_cl = cl.array.empty(queue, (frames * I,), dtype = np.float32)
    
    K_dense = np.zeros((I, frames), dtype = np.uint8)
    
    # needs to be updated for chunking
    # for now this is just the entire dataset

    # calculate gradient offsets 
    # Wt[t] = sum_i W[t, i]
    cl.enqueue_copy(queue, W_cl.data, np.ascontiguousarray(W.T))
    cl.enqueue_copy(queue, B_cl.data, np.ascontiguousarray(B.T))
    cl_code.calculate_Wt2(queue, (C,), None,
        W_cl.data, Wsums_cl.data, I, C)
        
    ts = np.zeros((frames, C), dtype = np.int32)
    Ts = np.zeros((frames,), dtype = np.int32)
    
    for dstart, dstop, dd in tqdm(chunk_csize(D, frames), desc = 'updating fluence estimates'):
        for i in tqdm(range(1), desc='setup', leave=False):
            K_dense.fill(0)
            for d in range(dstart, dstop):
                K_dense[inds[d], d - dstart] = K[d]
            cl.enqueue_copy(queue, K_cl.data, K_dense)

            cl.enqueue_copy(queue, P_cl.data, P[dstart: dstop])
            cl.enqueue_copy(queue, w_cl.data, w[dstart: dstop])
            cl.enqueue_copy(queue, b_cl.data, b[dstart: dstop])
             
            # g0[d] = sum_t P[d, t] sum_i W[t, i]
            cl_code.calculate_gw(queue, (dd,), None,
                P_cl.data, Wsums_cl.data, gw_cl.data, C)
            
            # make lookup table for significant classes as a function of frame id
            # ts[d, n] = n'th significant class id (t)
            # Ts[d]    = number of significant classes
            ts.fill(0)
            Ts.fill(0)
            # generate the frame lookup table 
            for d in range(dd):
                # good frame P[d, t] > Pmax[t]
                p = P[dstart + d] 
                t = np.where(p > (p.max() * tol_P))[0]
                ts[d, :t.shape[0]] = t
                Ts[d] = t.shape[0]
                #print(dstart + d, t)
            
            cl.enqueue_copy(queue, ts_cl.data, ts)
            cl.enqueue_copy(queue, Ts_cl.data, Ts)
            
            # background[i, d] = sum_l b[d, l] B[l, i]
            cl_code.calculate_background_id(queue, (dd, I), None,
                B_cl.data, b_cl.data, background_cl.data, L, I, frames)
         
        cl_code.update_w(queue, (dd,), None, 
            P_cl.data, K_cl.data, background_cl.data, 
            W_cl.data, w_cl.data, Ts_cl.data, ts_cl.data,
            gw_cl.data, minval, I, C, frames)
        
        cl.enqueue_copy(queue, w[dstart:dstop], w_cl.data)
    
        if update_b :
            for i in tqdm(range(1), desc = 'updating background weights'):
                # gb[l] = sum_i B[i, l]
                cl_code.calculate_gb(queue, (L,), None,
                    B_cl.data, gb_cl.data, I, L)
                
                cl_code.update_b(queue, (dd, L), None, 
                    P_cl.data, K_cl.data, background_cl.data, 
                    W_cl.data, w_cl.data, B_cl.data, b_cl.data, Ts_cl.data, ts_cl.data,
                    gb_cl.data, minval, I, C, frames, L)
                
                cl.enqueue_copy(queue, b[dstart:dstop], b_cl.data)



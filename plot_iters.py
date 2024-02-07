import pickle
import numpy as np
import utils
import glob
import os

class Empty():
    pass

fnams = np.sort(glob.glob('recon_*.pickle'))

for fnam in fnams:
    a = pickle.load(open(fnam, 'rb'))
    
    utils.plot_iter(a, a.iterations)
os.system("pdfunite recon_*.pdf recon.pdf")

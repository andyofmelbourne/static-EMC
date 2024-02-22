import pickle
import numpy as np
import utils
import glob
import os
import sys


c = utils.load_config(sys.argv[1])

class Empty():
    pass

fnams = np.sort(glob.glob('recon_*.pickle'))


for fnam in fnams:
    a = pickle.load(open(fnam, 'rb'))
    
    utils.plot_iter(c, a, a.iterations)
os.system("pdfunite recon_*.pdf recon.pdf")

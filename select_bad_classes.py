import argparse
from PyQt5 import QtGui, QtCore, QtWidgets
from collections import defaultdict
import pyqtgraph as pg
import numpy as np
import signal
import os
import pickle
from static_emc_init import A
import sys
import scipy.special

def get_classes_by_favour(r):
    C      = r.W.shape[0]
    favour = np.sum(r.P, axis=0)
    ts     = np.argsort(favour)[::-1]

    for t in range(r.C):
        print(t, ts[t], favour[ts[t]])

    return ts, assemble_classes(r, ts)

def assemble_classes(r, classes):
    image = np.empty(r.frame_shape, dtype=float)
    imshape = image[r.frame_slice].shape
    ims   = np.zeros((len(classes),) + imshape, dtype=float)
    for i, t in enumerate(classes) :
        image.fill(0)
        image.ravel()[r.pixel_indices] = r.W[t]
        ims[i] = image[r.frame_slice]
    return ims


# Get key mappings from Qt namespace
qt_keys = (
    (getattr(QtCore.Qt, attr), attr[4:])
    for attr in dir(QtCore.Qt)
    if attr.startswith("Key_")
)
keys_mapping = defaultdict(lambda: "unknown", qt_keys)


class Show_frames(QtWidgets.QMainWindow):
    def __init__(self, c, r, K, inds):
        super(Show_frames, self).__init__()
        
        # find favorout class for each frame 
        self.classes = r.most_likely_classes[-1]
        self.K = K
        self.inds = inds
        self.r = r

        self.get_frames(c)
        self.initUI()
    
    def get_frames(self, c):
        ds = np.where(self.classes == c)[0]
        image  = np.empty(self.r.frame_shape, dtype=float)
        imshape = image[self.r.frame_slice].shape
        frames = np.zeros((len(ds)+1,) + imshape, dtype=np.float32)
        k      = np.empty((self.r.I,), dtype=np.uint8)
        ksums  = []

        # test
        self.w = self.r.w[ds]
        
        # calculate log likelihood
        #LL = np.empty((len(ds),), dtype=float)
        for i, d in enumerate(ds) :
            k.fill(0)
            image.fill(0)
            
            ksums.append(np.sum(self.K[d]))
            k[self.inds[d]] = self.K[d]
            
            #T = self.r.w[d] * self.r.W[c] + self.r.b[d, 0] * self.r.B[0]
            #T = self.r.W[c]
            #LL1 = np.sum( k * np.log(T) - T - scipy.special.gammaln(k + 1) )
            #LL2 = np.sum( k * np.log(np.clip(k, 1, None)) - k - scipy.special.gammaln(k + 1) )
            #LL[i] = LL1 / LL2
            #LL[i] = LL1 - LL2
            #LL[i] = np.sum( k * np.log(T) - T - scipy.special.gammaln(k + 1) ) / np.sum(k)
            #LL[i] = self.r.P[d, c] * np.sum(k * np.log(T) - T - scipy.special.gammaln(k + 1) )
            #LL[i] = self.r.P[d, c] 
            
            image.ravel()[self.r.pixel_indices] = k
            
            frames[i] = image[self.r.frame_slice] 

        if c == 90 :
            LL, ds_LL = pickle.load(open('LLs_exclude.pickle', 'rb'))
            assert(np.allclose(ds_LL, ds))
            #LL -= np.log(np.array(ksums))
            i = np.argsort(LL)[::-1]
            self.LL = LL[i]
            self.ksums = np.array(ksums)[i]
        else :
            i = np.arange(len(ds))
            self.LL = np.ones(len(ds),)
            self.ksums = np.array(ksums)

        #print(np.sort(LL)[::-1])
        #i = np.argsort(ksums)[::-1]
        self.w  = self.w[i]
        self.ds = ds[i]
        frames[:-1] = frames[i]
        frames[-1] = np.mean(frames[:-1], axis=0)
        self.frames = frames
        
        print('{:>10} {:>10}'.format('class', 'frame index'))
        for d in ds[i]:
            print('{:>10} {:>10}'.format(c, d))

    def timeLineChanged(self):
        # draw border if class is "bad"
        index, time = self.plot.timeIndex(self.plot.timeLine)
        print('index', index, 'frame', self.ds[index], 'ksum', self.ksums[index], 'LL', self.LL[index], 'w', self.w[index])
        
    def initUI(self):
        # 2D plot for the cspad and mask
        self.plot = pg.ImageView()
        self.setCentralWidget(self.plot)
        
        # send signal when timeline changes value (class number)
        self.plot.timeLine.sigPositionChanged.connect(self.timeLineChanged)
        
        # display the image
        self.plot.setImage(self.frames, autoRange = True, autoLevels = False, levels = (0, 5), autoHistogramRange = True)
        
        ## Display the widget as a new window
        self.resize(800, 480)
        print('done')
        #self.show()

class Application(QtWidgets.QMainWindow):
    keyPressed = QtCore.pyqtSignal(QtCore.QEvent)
    
    def __init__(self, classes, W, r, K, inds):
        super().__init__()
        self.keyPressed.connect(self.on_key)
         
        self.W = W
        self.r = r
        self.K = K
        self.inds = inds
        self.bad = np.ones(W.shape[0], dtype=bool)
        self.classes = classes
        self.initUI()

    def keyPressEvent(self, event):
        #super(KeyPressWindow, self).keyPressEvent(event)
        super(Application, self).keyPressEvent(event)
        self.keyPressed.emit(event) 
    
    def initUI(self):
        # Define a top-level widget to hold everything
        #w = QtWidgets.QWidget()
        #self.w = KeyPressWindow()
        
        # 2D plot for the cspad and mask
        self.plot = pg.ImageView()
        self.imageitem = self.plot.getImageItem()

        # send signal when timeline changes value (class number)
        self.plot.timeLine.sigPositionChanged.connect(self.timeLineChanged)
        
        # Create a grid layout to manage the widgets size and position
        self.setCentralWidget(self.plot)
        #layout = QtWidgets.QGridLayout()
        #self.w.setLayout(layout)
        #self.w.keyPressed.connect(self.on_key)

        #layout.addWidget(self.plot, 0, 0)

        # display the image
        self.plot.setImage(self.W**0.5, autoRange = True, autoLevels = True, autoHistogramRange = True)
        self.timeLineChanged()
        
        ## Display the widget as a new window
        self.resize(800, 480)
        #self.show()
        

    def timeLineChanged(self):
        # draw border if class is "bad"
        index, time = self.plot.timeIndex(self.plot.timeLine)
        print('index', index, 'class', self.classes[index])
        if self.bad[index] == True:
            self.imageitem.setBorder('r')
        else :
            self.imageitem.setBorder('g')
    
    def on_key(self, event):
        if keys_mapping[event.key()] == 'X':
            index, time = self.plot.timeIndex(self.plot.timeLine)
            self.bad[index] = ~self.bad[index]
            print('setting index:', index, time, 'class:', self.classes[index], 'to', ~self.bad[index])
            self.timeLineChanged()
        
        # save list of good classes in recon
        elif keys_mapping[event.key()] == 'S':
            out = 'good_classes.pickle'
            print('saving list of good classes to:', out)
            #pickle.dump(~self.bad[np.array(self.classes)], open(out, 'wb'))
            t = np.zeros_like(self.bad)
            t[np.array(self.classes)[~self.bad]] = True
            pickle.dump(t, open(out, 'wb'))
            print('done')
            
        elif keys_mapping[event.key()] == 'K':
            index, time = self.plot.timeIndex(self.plot.timeLine)
            self.s = Show_frames(self.classes[index], self.r, self.K, self.inds)
            self.s.show()
        
        


if __name__ == '__main__':
    #fnam = 'recon_0020.pickle'
    fnam = sys.argv[1]
    r = pickle.load(open(fnam, 'rb'))
    K, inds = pickle.load(open('photons.pickle', 'rb'))
    #K, inds = [], []
    ts, ims = get_classes_by_favour(r)

    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    
    # Always start by initializing Qt (only once per application)
    app = QtWidgets.QApplication([])
        
    a = Application(ts, ims, r, K, inds)
    a.show()
    print('hello')

    ## Start the Qt event loop
    app.exec_()


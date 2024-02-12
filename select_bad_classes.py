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

def get_classes_by_favour(r):
    C      = r.W.shape[0]
    favour = np.sum(r.P, axis=0)
    ts     = np.argsort(favour)[::-1]

    for t in range(r.C):
        print(t, ts[t], favour[ts[t]])

    return ts, assemble_classes(r, ts)

def assemble_classes(r, classes):
    image = np.empty(r.frame_shape, dtype=float)
    ims   = np.zeros((len(classes), 128, 128), dtype=float)
    for i, t in enumerate(classes) :
        image.fill(0)
        image.ravel()[r.pixel_indices] = r.W[t]
        ims[i] = image[0, :128, :128]
    return ims


# Get key mappings from Qt namespace
qt_keys = (
    (getattr(QtCore.Qt, attr), attr[4:])
    for attr in dir(QtCore.Qt)
    if attr.startswith("Key_")
)
keys_mapping = defaultdict(lambda: "unknown", qt_keys)

class KeyPressWindow(QtWidgets.QWidget):
    keyPressed = QtCore.pyqtSignal(QtCore.QEvent)

    def __init__(self):
        super(KeyPressWindow, self).__init__()
        self.keyPressed.connect(self.on_key)

    def keyPressEvent(self, event):
        super(KeyPressWindow, self).keyPressEvent(event)
        self.keyPressed.emit(event) 
    
    def on_key(self, event):
        print(keys_mapping[event.key()])


class Application:
    def __init__(self, classes, W):
        self.W = W
        self.bad = np.ones(W.shape[0], dtype=bool)
        self.classes = classes
        self.initUI()

    def initUI(self):
        signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
        
        # Always start by initializing Qt (only once per application)
        self.app = QtWidgets.QApplication([])
        
        # Define a top-level widget to hold everything
        #w = QtWidgets.QWidget()
        self.w = KeyPressWindow()
        
        # 2D plot for the cspad and mask
        self.plot = pg.ImageView()
        self.imageitem = self.plot.getImageItem()

        # send signal when timeline changes value (class number)
        self.plot.timeLine.sigPositionChanged.connect(self.timeLineChanged)
        
        # Create a grid layout to manage the widgets size and position
        layout = QtWidgets.QGridLayout()
        self.w.setLayout(layout)
        self.w.keyPressed.connect(self.on_key)

        layout.addWidget(self.plot, 0, 0)

        # display the image
        self.plot.setImage(self.W**0.5, autoRange = False, autoLevels = False, autoHistogramRange = False)
        
        ## Display the widget as a new window
        self.w.resize(800, 480)
        self.w.show()
        
        ## Start the Qt event loop
        self.app.exec_()

    def timeLineChanged(self):
        # draw border if class is "bad"
        index, time = self.plot.timeIndex(self.plot.timeLine)
        if self.bad[index] == True:
            self.imageitem.setBorder('r')
        else :
            self.imageitem.setBorder('g')
    
    def on_key(self, event):
        if keys_mapping[event.key()] == 'X':
            index, time = self.plot.timeIndex(self.plot.timeLine)
            self.bad[index] = ~self.bad[index]
            print('setting index:', index, time, 'class:', self.classes[index], 'to', self.bad[index])
            self.timeLineChanged()
        
        # save list of good classes in recon
        elif keys_mapping[event.key()] == 'S':
            out = 'good_classes.pickle'
            print('saving list of good classes to:', out)
            #pickle.dump(~self.bad[np.array(self.classes)], open(out, 'wb'))
            t = np.zeros_like(self.bad)
            t[np.array(self.classes)[~self.bad]] = True
            pickle.dump(t, open(out, 'wb'))
            
        
        """
        elif keys_mapping[event.key()] == 'M':
            r = pickle.load(open('recon.pickle', 'rb'))
            if np.any(~self.scale) :
                bad_classes = np.array(self.classes)[~self.scale]
                bad_frames = []
                
                for d in range(r.D):
                    if np.argmax(r.P[d]) in bad_classes :
                        bad_frames.append(d)
                    
                print('found', len(bad_frames), 'bad frames')
                 
                # sort bad_frames to prevent messing with indices
                bad_frames = np.sort(bad_frames)[::-1]
                
                # remove bad frames
                for d in bad_frames:
                    r.K.pop(d)
                    r.inds.pop(d)
                
                r.w = np.delete(r.w, bad_frames)
                r.b = np.delete(r.b, bad_frames, axis = 0)
                r.P  = np.delete(r.P, bad_frames, axis = 0)
                r.LR = np.delete(r.LR, bad_frames, axis = 0)
            
                r.D = r.w.shape[0]

            # re-intitialise?
            print('reinitialising arrays')
            r.w[:] = np.ones_like(r.w)
            r.b[:] = np.zeros_like(r.b)
            r.W[:] = np.random.random(r.W.shape)
            
            pickle.dump(r, open('recon.pickle', 'wb'))
            print('done')
        """
        


if __name__ == '__main__':
    #fnam = 'recon_0020.pickle'
    fnam = sys.argv[1]
    r = pickle.load(open(fnam, 'rb'))
    ts, ims = get_classes_by_favour(r)
    Application(ts, ims)
    

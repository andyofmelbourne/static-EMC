import numpy as np
import h5py
import pickle
import config as c
import sys

# load recon file
recon = sys.argv[1]
r = pickle.load(open(recon, 'rb'))

# load list of good classes
gc = sys.argv[2]
good_classes = pickle.load(open(gc, 'rb'))

# get frames that have good classes as their most favoured class
good_frames = good_classes[r.most_likely_classes[-1]]

classes_frames = np.array(r.most_likely_classes[-1])

good_class_labels = np.where(good_classes)[0]

background_weights = r.b

# write to cxi files
for i, fnam in enumerate(c.data) :
    with h5py.File(fnam, 'a') as f:
        D = f['entry_1/data_1/data'].shape[0]
        good_frames_fnam = np.zeros((D,), dtype=bool)
        classes_fnam = np.zeros((D,), dtype=np.uint16)
        background_weights_fnam = np.zeros((D, r.b.shape[1]), dtype=np.float32)

        # invalid class id
        classes_fnam[:] = np.iinfo(classes_fnam.dtype).max
        
        # location of frame indices for this file
        j = np.where(np.array(r.file_index) == i)[0]
        
        # frame indices for this file
        frame_index = np.array(r.frame_index)[j]
        
        good_frames_fnam[frame_index] = good_frames[j]
        classes_fnam[frame_index]     = classes_frames[j]
        background_weights_fnam[frame_index]     = background_weights[j]

        print()
        print('writing', np.sum(good_frames_fnam), 'good frames to', fnam, '/static_emc/good_hit')
        if 'static_emc/good_hit' in f :
            f['static_emc/good_hit'][:] = good_frames_fnam
        else :
            f['static_emc/good_hit'] = good_frames_fnam

        print('writing', classes_fnam.shape[0], 'class labels to', fnam, '/static_emc/class')
        if 'static_emc/class' in f :
            f['static_emc/class'][:] = classes_fnam
        else :
            f['static_emc/class'] = classes_fnam

        print('writing',good_class_labels.shape[0], 'list of good class labels to ', fnam, '/static_emc/good_classes')
        if 'static_emc/good_classes' in f :
            del f['static_emc/good_classes']
        f['static_emc/good_classes'] = good_class_labels
    
        print('writing',background_weights_fnam.shape, 'background weights to ', fnam, '/static_emc/background_weights')
        if 'static_emc/background_weights' in f :
            del f['static_emc/background_weights']
        f['static_emc/background_weights'] = background_weights_fnam


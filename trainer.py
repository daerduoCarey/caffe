import numpy as np
import matplotlib.pyplot as plt
import caffe
from pylab import *

caffe_root = './'

niter = 200
# losses will also be stored in the log
train_loss = np.zeros(niter)

caffe.set_device(0)
caffe.set_mode_gpu()
# We create a solver that fine-tunes from a previously trained network.
solver = caffe.SGDSolver(caffe_root + 'models/finetune_flickr_style/solver.prototxt')
solver.net.copy_from(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

# We run the solver for niter times, and record the training loss.
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    if it % 10 == 0:
        print 'iter %d, finetune_loss=%f' % (it, train_loss[it])
print 'done'

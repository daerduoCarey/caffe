import numpy as np
import matplotlib.pyplot as plt
import caffe
from pylab import *
import vis_utils as vu

def data_unit(net, file_name):

	n, c, h, w = net.blobs['data'].data.shape

	plt.subplot(131)
        plt.title('Original Image')
        plt.axis('off')
	vu.visualize_one_channel_images(net.blobs['data'].data.reshape(n, h, w))

	plt.subplot(132)
        plt.title('ST Output')
        plt.axis('off')
	vu.visualize_one_channel_images(net.blobs['st_output'].data.reshape(n, h, w))

	plt.subplot(133)
        plt.axis('off')
        plt.title('Correctness')

	acc = np.zeros((n, h, w, 3))

	gt_label = net.blobs['label'].data
	est_label = np.argmax(net.blobs['class'].data, axis=1)
	err = (est_label <> gt_label)
        ind = np.array(range(n))[err]
	for i in ind:
		acc[i] = np.ones((h, w, 3))

	plt.imshow(vu.vis_grid(acc))
	plt.gca().axis('off')

	plt.savefig(file_name+'.jpg', dpi = 1000)
        plt.close()


caffe_root = './'

niter = 10000
display = 1
# losses will also be stored in the log
train_loss = np.zeros(niter)

caffe.set_device(0)
caffe.set_mode_gpu()
# We create a solver that fine-tunes from a previously trained network.
solver = caffe.SGDSolver(caffe_root + 'examples/mnist_tests/ST_CNN_RST/solver.prototxt')

# We run the solver for niter times, and record the training loss.
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    if it % display == 0:
        print 'iter %d, finetune_loss=%f' % (it, train_loss[it])
        data_unit(solver.net, 'logs/'+str(it))
        print solver.net.blobs['theta'].diff[0]
        print solver.net.blobs['theta'].data[0]
print 'done'

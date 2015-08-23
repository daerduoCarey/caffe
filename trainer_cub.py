import numpy as np
import matplotlib.pyplot as plt
import caffe
from pylab import *
import vis_utils as vu

def data_unit(net, file_name):

	n, c, h, w = net.blobs['data'].data.shape

	plt.subplot(221)
        plt.title('Original Image')
        plt.axis('off')
	vu.vis_square(net.blobs['data'].data.transpose(0, 2, 3, 1))

	plt.subplot(223)
        plt.title('Inc1/data')
        plt.axis('off')
	vu.vis_square(net.blobs['inc1/data'].data.transpose(0, 2, 3, 1))

	plt.subplot(224)
        plt.title('Inc2/data')
        plt.axis('off')
	vu.vis_square(net.blobs['inc2/data'].data.transpose(0, 2, 3, 1))

	plt.subplot(222)
        plt.axis('off')
        plt.title('Correctness')

	acc = np.zeros((n, h, w, 3))

	gt_label = net.blobs['label'].data
	est_label = np.argmax(net.blobs['final/res'].data, axis=1)
	err = (est_label <> gt_label)
        ind = np.array(range(n))[err]
	for i in ind:
		acc[i] = np.ones((h, w, 3))

	plt.imshow(vu.vis_grid(acc))
	plt.gca().axis('off')

	plt.savefig(file_name+'.jpg', dpi = 1000)
        plt.close()


caffe_root = './'

niter = 100000
display = 10
# losses will also be stored in the log
train_loss = np.zeros(niter)

caffe.set_device(0)
caffe.set_mode_gpu()
# We create a solver that fine-tunes from a previously trained network.
solver = caffe.SGDSolver(caffe_root + 'models/CUB_googLeNet_ST/solver.prototxt')
solver.net.copy_from(caffe_root + 'models/CUB_googLeNet_ST/init_CUB_googLeNet_ST_NO_ST_USED_WITH_WEIGHT_DECAY.caffemodel')

# We run the solver for niter times, and record the training loss.
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    if it % display == 0:
        print 'iter %d, finetune_loss=%f' % (it, train_loss[it])
        #data_unit(solver.net, 'logs/'+str(it))
        #print 'st/final_ip2'
        #print solver.net.blobs['st/final_ip2'].data[0]
        #print 'weight'
        #print solver.net.params['st/theta_1'][0].data[0]
        #print 'bias'
        #print solver.net.params['st/theta_1'][1].data[0]
        #print 'theta'
        print solver.net.blobs['st/theta_1'].data[0]
        print solver.net.blobs['st/theta_2'].data[0]
        #print 'dtheta'
        #print solver.net.blobs['st/theta_1'].diff[0]
        #print 'dweight'
        #print solver.net.params['st/theta_1'][0].diff[0]
    if it % 100 == 0:
        data_unit(solver.net, 'logs/'+str(it))
print 'done'

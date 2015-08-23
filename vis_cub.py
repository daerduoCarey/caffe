import numpy as np
import matplotlib.pyplot as plt
import caffe
from math import ceil, sqrt

import vis_utils as vu

def data_unit(net, file_name):

	n, c, h, w = net.blobs['data'].data.shape

 	f = open(file_name+'.txt', 'w')

	plt.subplot(221)
        plt.title('Original Image')
        plt.axis('off')
	vu.vis_square(net.blobs['data'].data.transpose(0, 2, 3, 1))

	plt.subplot(223)
        plt.title('Inc1/data')
        plt.axis('off')
        print net.blobs['st/theta_1'].data
	vu.vis_square(net.blobs['inc1/data'].data.transpose(0, 2, 3, 1))

	plt.subplot(224)
        plt.title('Inc2/data')
        plt.axis('off')
        print net.blobs['st/theta_2'].data
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
		x = i/ceil(sqrt(n))
		y = i%ceil(sqrt(n))
		f.write('Bird at (%d, %d) should be %d, but is classified as %d\n'%(x, y, gt_label[i], est_label[i]))
		acc[i] = np.ones((h, w, 3))

	plt.imshow(vu.vis_grid(acc))
	plt.gca().axis('off')

	plt.savefig(file_name+'.jpg', dpi = 1000)
	plt.close()

def main():
	
	caffe_root = './'
	res_root = 'res/'
	
	tot = 10

        caffe.set_device(1)
	caffe.set_mode_gpu()
	net = caffe.Net(caffe_root + 'models/CUB_googLeNet_ST/train_test.prototxt',
			caffe_root + 'models/CUB_googLeNet_ST/caffemodels/CUB_googLeNet_ST_PRETRAINED_ST_USED_WITH_WEIGHT_DECAY_iter_100000.caffemodel',
			caffe.TEST)

	for i in xrange(tot):
		print '%d/%d' % (i, tot)
		net.forward()
		data_unit(net, res_root+'res'+'{:08}'.format(i))

main()

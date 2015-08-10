import numpy as np
import matplotlib.pyplot as plt
import caffe
from math import ceil, sqrt

import vis_utils as vu

def data_unit(net, file_name):

	n, c, h, w = net.blobs['data'].data.shape

 	f = open(file_name+'.txt', 'w')

	plt.subplot(131)
	vu.visualize_one_channel_images(net.blobs['data'].data.reshape(n, h, w))

	plt.subplot(132)
	vu.visualize_one_channel_images(net.blobs['st_output'].data.reshape(n, h, w))

	plt.subplot(133)
	acc = np.zeros((n, h, w, 3))

	gt_label = net.blobs['label'].data
	est_label = np.argmax(net.blobs['class'].data, axis=1)
	err = (est_label <> gt_label)
	ind = np.array(range(n))[err]
	for i in ind:
		x = i/ceil(sqrt(n))
		y = i%ceil(sqrt(n))
		f.write('Digit at (%d, %d) should be %d, but is classified as %d\n'%(x, y, gt_label[i], est_label[i]))
		acc[i] = np.ones((h, w, 3))

	plt.imshow(vu.vis_grid(acc))
	plt.gca().axis('off')

	plt.savefig(file_name+'.jpg', dpi = 100)
	plt.close()

def main():
	
	caffe_root = './'
	res_root = 'res/'
	
	tot = 10

	caffe.set_mode_cpu()
	net = caffe.Net(caffe_root + 'examples/mnist_tests/ST_CNN_RST/ST_CNN.prototxt',
			caffe_root + 'examples/mnist_tests/ST_CNN_RST/ST_CNN_iter_100000.caffemodel',
			caffe.TEST)

	for i in xrange(tot):
		print '%d/%d' % (i, tot)
		net.forward()
		data_unit(net, res_root+'res'+'{:08}'.format(i))

main()

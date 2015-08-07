import numpy as np
import matplotlib.pyplot as plt
import caffe

import vis_utils as vu

def data_unit(net, file_name):

	n, c, h, w = net.blobs['data'].data.shape

	plt.subplot(121)
	vu.visualize_one_channel_images(net.blobs['data'].data.reshape(n, h, w))

	plt.subplot(122)
	vu.visualize_one_channel_images(net.blobs['st_output'].data.reshape(n, h, w))

	plt.savefig(file_name, dpi = 1000)
	plt.close()

def main():
	
	caffe_root = './'
	res_root = 'res/'
	
	tot = 10

	caffe.set_mode_cpu()
	net = caffe.Net(caffe_root + 'examples/mnist_st/lenet_conv_st.prototxt',
			caffe_root + 'examples/mnist_st/lenet_st_iter_90000.caffemodel',
			caffe.TEST)

	for i in xrange(tot):
		print '%d/%d' % (i, tot)
		net.forward()
		data_unit(net, res_root+'res'+'{:08}'.format(i)+'.jpg')

main()

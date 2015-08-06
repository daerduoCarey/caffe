import numpy as np
import matplotlib.pyplot as plt
import caffe

import vis_utils as vu

caffe_root = './'

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'examples/mnist_st/lenet_conv_st.prototxt',
		caffe_root + 'examples/mnist_st/lenet_st_iter_90000.caffemodel',
		caffe.TEST)

net.forward()
net.forward()
net.forward()

n, c, h, w = net.blobs['data'].data.shape

print net.blobs['data'].data.shape

vu.visualize_one_channel_images(net.blobs['data'].data.reshape(n, h, w))
vu.visualize_one_channel_images(net.blobs['st_output'].data.reshape(n, h, w))

print net.blobs['theta'].data.reshape(n, 2, 3)

gt_label = np.int32(net.blobs['label'].data)
est_label = np.argmax(net.blobs['ip2'].data, axis = 1)

for i in range(gt_label.size):
	print '%d, should be %d, is classified as %d' % (i, gt_label[i], est_label[i]), 
	if gt_label[i] != est_label[i]:
		print '   DIFF'
	else:
		print

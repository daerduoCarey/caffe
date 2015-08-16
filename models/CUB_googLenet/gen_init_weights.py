import caffe
import numpy as np

caffe_root = '../../'

net = caffe.Net(caffe_root + 'models/bvlc_googlenet/deploy.prototxt', 
		caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel', 
		caffe.TEST)

cub_net = caffe.Net(caffe_root + 'models/CUB_googLeNet/train_test.prototxt', 
		    caffe_root + 'models/'

print net.params.keys()

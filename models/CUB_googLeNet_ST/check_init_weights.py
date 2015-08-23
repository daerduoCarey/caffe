import caffe
import numpy as np

caffe_root = '../../'

net = caffe.Net(caffe_root + 'models/bvlc_googlenet/deploy.prototxt', 
		caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel', 
		caffe.TEST)

cub_net = caffe.Net(caffe_root + 'models/CUB_googLeNet_ST/deploy.prototxt', 
                    caffe_root + 'models/CUB_googLeNet_ST/init_CUB_googLeNet.caffemodel', 
		    caffe.TEST)

net_params = net.params.keys()
prefix_list = ['inc1', 'inc2', 'st']

for param in net_params[:-1]:
    for idx, value in enumerate(net.params[param]):
        for prefix in prefix_list:
            print 'Comparing to %s, %d' % ('/'.join([prefix, param]), idx)
            print np.sum(cub_net.params['/'.join([prefix, param])][idx].data <> value.data)
           
print cub_net.params['cub/st/theta_1'][0].data
print cub_net.params['cub/st/theta_1'][1].data

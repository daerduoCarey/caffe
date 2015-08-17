import caffe
import numpy as np
import matplotlib.pyplot as plt 
import vis_utils as vu

def draw(data_inc1, data_inc2, file_name):
    plt.subplot(1, 2, 1)
    plt.title('data_inc1')
    vu.vis_square(data_inc1.transpose(0, 2, 3, 1))
    plt.subplot(1, 2, 2)
    plt.title('data_inc2')
    vu.vis_square(data_inc2.transpose(0, 2, 3, 1))
    plt.savefig(file_name)
    plt.close()

caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('models/CUB_googLeNet_ST/solver.prototxt')
net = solver.net

net.copy_from('models/CUB_googLeNet_ST/init_CUB_googLeNet.caffemodel')

niter = 1000
train_loss = np.zeros(niter)

display = 1

net.forward()
draw(net.blobs['inc1/data'].data, net.blobs['inc2/data'].data, 'res/init.jpg')

for it in range(niter):
    
    solver.step(1)
    train_loss[it] = net.blobs['loss'].data
    
    if it % display == 0:
        print 'Iter %d: loss = %f' % (it, train_loss[it])
        data_inc1 = net.blobs['inc1/data'].data
        data_inc2 = net.blobs['inc2/data'].data
        draw(data_inc1, data_inc2, 'res/iter='+str(it)+'.jpg')

        print net.blobs['st/theta_1'].data

import numpy as np
import scipy.misc as sm
import lmdb
import caffe

import vis_utils as vu
import st_layers as st

def parse_value(raw_datum):

	datum = caffe.proto.caffe_pb2.Datum()
	datum.ParseFromString(raw_datum)

	flat_x = np.fromstring(datum.data, dtype=np.uint8)
	x = flat_x.reshape(datum.channels, datum.height, datum.width)
	y = datum.label

	return x, y

def distortion(image):

	n, c, h, w = image.shape

	Theta = np.zeros((n, 2, 3))
	for i in xrange(n):
		scale = np.random.rand() * 0.7 + 0.3;
		degree = np.random.rand() * np.pi * 2;
		
		shift_x = np.random.rand() * 0.4 - 0.2
		shift_y = np.random.rand()  * 0.4 - 0.2

		shift_to_center = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])	
		scale_and_rotate = np.array([[scale * np.sin(degree), -scale * np.cos(degree), 0], 
				     [scale * np.cos(degree), scale * np.sin(degree), 0], [0, 0, 1]])
		shift_back = np.array([[1, 0, 0.5 + shift_x], [0, 1, 0.5 + shift_y], [0, 0, 1]])
		
		res = np.linalg.inv(np.dot(shift_back, np.dot(scale_and_rotate, shift_to_center)))

		Theta[i] = res[0:2]

	return st.st_forward(Theta, image, h, w)[0]

def main():

	env = lmdb.open('mnist_train_lmdb', readonly=True)

	with env.begin() as txn:
		cursor = txn.cursor()
		for key, value in cursor:
			(image, label) = parse_value(value)
			if(np.random.rand() < 0.9):
				print 'examples/mnist/mnist_train_images/' + key + '.jpg', label	
			else:
				print 'examples/mnist/mnist_distorted_train_images/' + key + '.jpg', label

main()

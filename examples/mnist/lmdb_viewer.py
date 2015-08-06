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
		scale = np.random.rand() * 0.3 + 0.7;
		degree = np.random.rand() * np.pi / 2 - np.pi / 4;
		
		shift_x = np.random.rand() * 0.4 - 0.2
		shift_y = np.random.rand()  * 0.4 - 0.2

		shift_to_center = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])	
		scale_and_rotate = np.array([[scale * np.cos(degree), -scale * np.sin(degree), 0], 
				     [scale * np.sin(degree), scale * np.cos(degree), 0], [0, 0, 1]])
		shift_back = np.array([[1, 0, 0.5 + shift_x], [0, 1, 0.5 + shift_y], [0, 0, 1]])
		
		res = np.linalg.inv(np.dot(shift_back, np.dot(scale_and_rotate, shift_to_center)))

		Theta[i] = res[0:2]

	return st.st_forward(Theta, image, h, w)[0]

def main():

	env = lmdb.open('mnist_test_lmdb', readonly=True)

	count = 0
	with env.begin() as txn:
		cursor = txn.cursor()
		for key, value in cursor:
			count = count + 1
			(image, label) = parse_value(value)
			c,h,w = image.shape
			image = np.reshape(image, (1,1,h,w))
			image = distortion(image)
			sm.imsave('mnist_distorted2_test_images/'+key+'.jpg', image.reshape(h, w))
			if(count%100==0):
				print count

main()

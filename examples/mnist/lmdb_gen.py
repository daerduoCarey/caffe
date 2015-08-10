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

	scale = 1
	degree = 0
	shift_x = 0
	shift_y = 0

	for i in xrange(n):
		#scale = np.random.rand() * 0.3 + 0.7;
		degree = np.random.rand() * np.pi / 2 - np.pi / 4;
		#shift_x = np.random.rand() * 0.4 - 0.2
		#shift_y = np.random.rand()  * 0.4 - 0.2

		srt = np.array([[scale * np.cos(degree), -scale * np.sin(degree), shift_x], 
				     [scale * np.sin(degree), scale * np.cos(degree), shift_y], [0, 0, 1]])
		
		res = np.linalg.inv(srt)

		Theta[i] = res[0:2]

	return st.st_forward(Theta, image, h, w)[0]

def main():

	env = lmdb.open('mnist_train_lmdb', readonly=True)
	map_size = 100000000
	env_out = lmdb.open('mnist_r_train_lmdb', map_size = map_size)
	count = 0
	with env.begin() as txn:
		cursor = txn.cursor()
		for key, value in cursor:
			(image, label) = parse_value(value)
			c,h,w = image.shape
			image = np.reshape(image, (1,1,h,w))
			image = distortion(image)
			image = np.reshape(image, (h, w))
			
			sm.imsave('mnist_r_train_images/'+key+'.jpg', image)
			
			datum = caffe.proto.caffe_pb2.Datum()
			datum.channels = 1
			datum.height = h
			datum.width = w
			datum.data = np.uint8(image).tostring()
			datum.label = int(label)
			str_id = '{:08}'.format(count)
			with env_out.begin(write=True) as txn:
				txn.put(str_id.encode('ascii'), datum.SerializeToString())

			count = count + 1
			if(count % 100 == 0):
				print count


main()

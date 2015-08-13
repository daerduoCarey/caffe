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

def main():

	TRAIN_OR_TEST = 'test'

	env = lmdb.open('mnist_'+TRAIN_OR_TEST+'_lmdb', readonly=True)

	with env.begin() as txn:
		cursor = txn.cursor()
		for key, value in cursor:
			(image, label) = parse_value(value)
			print 'examples/mnist/mnist_r_'+TRAIN_OR_TEST+'_images/'+key+'.jpg', label

main()

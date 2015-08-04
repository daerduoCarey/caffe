import numpy as np
import lmdb
import caffe

import vis_utils as vu

def parse_value(raw_datum):

	datum = caffe.proto.caffe_pb2.Datum()
	datum.ParseFromString(raw_datum)

	flat_x = np.fromstring(datum.data, dtype=np.float64)
	print datum.data
	print datum.channels, ' ', datum.height, ' ', datum.width
	x = flat_x.reshape(datum.channels, datum.height, datum.width)
	y = datum.label

	return x, y

def main():

	env = lmdb.open('st_output', readonly=True)

	with env.begin() as txn:
		cursor = txn.cursor()
		for key, value in cursor:
			print 'key = ', key
			(image, label) = parse_value(value)
			print 'lable = ', label
			print 'image = '
			c, h, w = image.shape;
			image.reshape(1, c, h, w)
			vu.visualize_one_channel_images(image)

main()

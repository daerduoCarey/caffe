f1 = open('images.txt')
f2 = open('image_class_labels.txt')
f3 = open('train_test_split.txt')

prefix = "data/CUB_200_2011/images/"

for x, y, z in zip(f1.readlines(), f2.readlines(), f3.readlines()):
	no, file_name = x.split()
	no, label = y.split()
	no, train = z.split()
	if(int(train)==0):
		print prefix + file_name, label


#!/usr/bin/env sh

./build/tools/caffe test -model models/CUB_googLeNet_Mask/train_test.prototxt -weights models/CUB_googLeNet_Mask/caffemodels/CUB_googLeNet_Mask_iter_6000.caffemodel -gpu 0 -iterations 180

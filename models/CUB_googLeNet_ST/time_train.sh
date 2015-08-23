#!/usr/bin/env sh

build/tools/caffe time -model models/CUB_googLeNet_ST/train_test.prototxt -weights models/CUB_googLeNet_ST/init2_CUB_googLeNet.caffemodel -gpu 1 -iterations 5

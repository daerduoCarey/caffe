#!/usr/bin/env sh

./build/tools/caffe time -model models/CUB_googLeNet/train_test.prototxt -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel -gpu 0 -iterations 5

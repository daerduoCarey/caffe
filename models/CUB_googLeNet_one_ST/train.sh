#!/usr/bin/env sh

build/tools/caffe train -solver models/CUB_googLeNet_one_ST/solver.prototxt -weights models/CUB_googLeNet_one_ST/init.caffemodel -gpu 0

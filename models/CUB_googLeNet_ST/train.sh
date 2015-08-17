#!/usr/bin/env sh

build/tools/caffe train -solver models/CUB_googLeNet_ST/solver.prototxt -weights models/CUB_googLeNet_ST/init2_CUB_googLeNet.caffemodel -gpu 0

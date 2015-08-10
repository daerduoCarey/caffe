#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/st_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class HardSpatialTransformerLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  HardSpatialTransformerLayerTest()
 	 : blob_U_(new Blob<Dtype>(5, 3, 10, 10)),
 	   blob_theta_(new Blob<Dtype>(5, 2, 3, 1)),
 	   blob_V_(new Blob<Dtype>(5, 3, 10, 10)){

	  FillerParameter filler_param;
	  GaussianFiller<Dtype> filler(filler_param);
	  filler.Fill(this->blob_U_);
	  filler.Fill(this->blob_theta_);

	  vector<int> shape_theta;
	  shape_theta[0] = 5; shape_theta[1] = 6;
	  blob_theta_->Reshape(shape_theta);

	  blob_bottom_vec_.push_back(blob_U_);
	  blob_bottom_vec_.push_back(blob_theta_);
	  blob_top_vec_.push_back(blob_V_);
  }
  virtual ~HardSpatialTransformerLayerTest() { delete blob_V_; delete blob_theta_; delete blob_U_; }
  Blob<Dtype>* blob_U_;
  Blob<Dtype>* blob_theta_;
  Blob<Dtype>* blob_V_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(HardSpatialTransformerLayerTest, TestDtypesAndDevices);

TYPED_TEST(HardSpatialTransformerLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    SpatialTransformerLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-4, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe

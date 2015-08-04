#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/st_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "fstream"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class SpatialTransformerLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SpatialTransformerLayerTest() {

	  std::ifstream fin("image_input.txt");

	  blob_U_ = new Blob<Dtype>();
	  blob_V_ = new Blob<Dtype>();
	  blob_theta_ = new Blob<Dtype>();

	  blob_U_->Reshape(1, 1, 4, 4);
	  blob_V_->Reshape(1, 1, 4, 4);
	  vector<int> theta_shape(3);
	  theta_shape[0] = 1; theta_shape[1] = 2; theta_shape[2] = 3;
	  blob_theta_->Reshape(theta_shape);

	  Dtype tmp;

	  Dtype* U = blob_U_->mutable_cpu_data();
	  for(int i=0; i<4; ++i)
		  for(int j=0; j<4; ++j) {
			  fin >> tmp;
			  U[blob_U_->offset(0, 0, i, j)] = tmp;
		  }

	  Dtype* theta = blob_theta_->mutable_cpu_data();
	  for(int i=0; i<2; ++i)
		  for(int j=0; j<3; ++j) {
			  fin >> tmp;
			  theta[blob_theta_->offset(0, i, j)] = tmp;
		  }

	  blob_bottom_vec_.push_back(blob_U_);
	  blob_bottom_vec_.push_back(blob_theta_);
	  blob_top_vec_.push_back(blob_V_);
  }
  virtual ~SpatialTransformerLayerTest() { delete blob_V_; delete blob_theta_; delete blob_U_; }
  Blob<Dtype>* blob_U_;
  Blob<Dtype>* blob_theta_;
  Blob<Dtype>* blob_V_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SpatialTransformerLayerTest, TestDtypesAndDevices);

//TYPED_TEST(SpatialTransformerLayerTest, TestSetUp) {
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  shared_ptr<SpatialTransformerLayer<Dtype> > layer(
//      new SpatialTransformerLayer<Dtype>(layer_param));
//  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//  EXPECT_EQ(this->blob_V_->num(), 1);
//  EXPECT_EQ(this->blob_V_->height(), 28);
//  EXPECT_EQ(this->blob_V_->width(), 28);
//  EXPECT_EQ(this->blob_V_->channels(), 1);
//}

TYPED_TEST(SpatialTransformerLayerTest, TestForwardAndBackward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {

	std::ofstream fout("image_log.txt");

	// Setup and Forward
	fout << "Setup and Forward" << std::endl << std::endl;

	LayerParameter layer_param;
    shared_ptr<SpatialTransformerLayer<Dtype> > layer(
        new SpatialTransformerLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    fout << "blob_U_->data:" << std::endl;
    for(int i=0; i<4; ++i) {
    	for(int j=0; j<4; ++j)
    		fout << this->blob_U_->cpu_data()[4*i+j] << "\t";
    	fout << std::endl;
    }

    fout << "blob_theta_->data:" << std::endl;
    for(int i=0; i<2; ++i) {
    	for(int j=0; j<3; ++j)
    		fout << this->blob_theta_->cpu_data()[3*i+j] << "\t";
    	fout << std::endl;
    }

    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    fout << "blob_V_->data:" << std::endl;
    for(int i=0; i<4; ++i) {
    	for(int j=0; j<4; ++j)
    		fout << this->blob_V_->cpu_data()[4*i+j] << "\t";
    	fout << std::endl;
    }

    // Give loss and Backward
    fout << std::endl << "Give loss and Backward" << std::endl << std::endl;

    Dtype* diff = this->blob_V_->mutable_cpu_diff();
    for(int i=0; i<4; ++i)
    	for(int j=0; j<4; ++j)
    		diff[this->blob_V_->offset(0, 0, i, j)] = this->blob_V_->cpu_data()[this->blob_V_->offset(0, 0, i, j)];

    fout << "blob_V_->diff:" << std::endl;
    for(int i=0; i<4; ++i) {
    	for(int j=0; j<4; ++j)
    		fout << this->blob_V_->cpu_diff()[4*i+j] << "\t";
    	fout << std::endl;
    }

    layer->Backward(this->blob_top_vec_, vector<bool>(2, true), this->blob_bottom_vec_);

    fout << "blob_theta_->diff:" << std::endl;
    for(int i=0; i<2; ++i) {
    	for(int j=0; j<3; ++j)
    		fout << this->blob_theta_->cpu_diff()[3*i+j] << "\t";
    	fout << std::endl;
    }

    fout << "blob_U_->diff:" << std::endl;
    for(int i=0; i<4; ++i) {
    	for(int j=0; j<4; ++j)
    		fout << this->blob_U_->cpu_diff()[4*i+j] << "\t";
    	fout << std::endl;
    }

  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(SpatialTransformerLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
    SpatialTransformerLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-4, 0);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe

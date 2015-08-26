#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/mask_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MaskTransformerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tMask Transformer Layer:: LayerSetUp: \t";

	if(this->layer_param_.mask_param().to_compute_du()) {
		to_compute_dU_ = true;
	}
}

template <typename Dtype>
void MaskTransformerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tMask Transformer Layer:: Reshape: \t";
	
	N = bottom[0]->shape(0);
	C = bottom[0]->shape(1);
	H = bottom[0]->shape(2);
	W = bottom[0]->shape(3);

	// reshape the parameter mm
	CHECK(bottom[1]->count(1) == H * W) << "The dimension of mask matrix and U is not the same!" << std::endl;
	CHECK(bottom[1]->shape(0) == N) << "The first dimension of mask matrix and U should be the same" << std::endl;

	// reshape V
	vector<int> shape(4);

	shape[0] = N;
	shape[1] = C;
	shape[2] = H;
	shape[3] = W;

	top[0]->Reshape(shape);
}

template <typename Dtype>
void MaskTransformerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	string prefix = "\t\tMask Transformer Layer:: Forward_cpu: \t";

	if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

	const Dtype* U = bottom[0]->cpu_data();
	const Dtype* mm = bottom[1]->cpu_data();
	Dtype* V = top[0]->mutable_cpu_data();

	caffe_set(top[0]->count(), (Dtype)0, V);

	int row_idx, idx;
	Dtype value;

	for(int i = 0; i < N; ++i) 
		row_idx = i * (H*W);
		for(int j = 0; j < H*W; ++j) {
			value = mm[row_idx + j];
			idx = row_idx * C + j;
			if(value >= 0 && value <= 1) {
				V[idx + 0 * (H*W)] = value * U[idx + 0 * (H*W)];
				V[idx + 1 * (H*W)] = value * U[idx + 1 * (H*W)];
				V[idx + 2 * (H*W)] = value * U[idx + 2 * (H*W)];
			} else if(value > 1) {
				V[idx + 0 * (H*W)] = U[idx + 0 * (H*W)];
				V[idx + 1 * (H*W)] = U[idx + 1 * (H*W)];
				V[idx + 2 * (H*W)] = U[idx + 2 * (H*W)];
			}
		}

	if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

template <typename Dtype>
void MaskTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

		string prefix = "\t\tMask Transformer Layer:: Backward_cpu: \t";

		if(global_debug) std::cout<<prefix<<"Starting!"<<std::endl;

		const Dtype* dV = top[0]->cpu_diff();
		const Dtype* U = bottom[0]->cpu_data();
		const Dtype* mm = bottom[1]->cpu_data();

		Dtype* dU = bottom[0]->mutable_cpu_diff();
		Dtype* dmm = bottom[1]->mutable_cpu_diff();

		caffe_set(bottom[0]->count(), (Dtype)0, dU);
		caffe_set(bottom[1]->count(), (Dtype)0, dmm);

		int row_idx, idx;
		Dtype value;

		for(int i = 0; i < N; ++i) 
			row_idx = i * (H*W);
			for(int j = 0; j < H*W; ++j) {
				value = mm[row_idx + j];
				idx = row_idx * C + j;
				if(value >= 0 && value <= 1) {
					dmm[row_idx + j] += U[idx + 0 * (H*W)] * dV[idx + 0 * (H*W)];
					dmm[row_idx + j] += U[idx + 1 * (H*W)] * dV[idx + 1 * (H*W)];
					dmm[row_idx + j] += U[idx + 2 * (H*W)] * dV[idx + 2 * (H*W)];
				}
			}

		if(to_compute_dU_) {
			for(int i = 0; i < N; ++i) 
				row_idx = i * (H*W);
				for(int j = 0; j < H*W; ++j) {
					value = mm[row_idx + j];
					idx = row_idx * C + j;
					if(value >= 0 && value <= 1) {
						dU[idx + 0 * (H*W)] = value * dV[idx + 0 * (H*W)];
						dU[idx + 1 * (H*W)] = value * dV[idx + 1 * (H*W)];
						dU[idx + 2 * (H*W)] = value * dV[idx + 2 * (H*W)];
					} else if(value > 1) {
						dU[idx + 0 * (H*W)] = dV[idx + 0 * (H*W)];
						dU[idx + 1 * (H*W)] = dV[idx + 1 * (H*W)];
						dU[idx + 2 * (H*W)] = dV[idx + 2 * (H*W)];
					}
				}
		}

		if(global_debug) std::cout<<prefix<<"Finished."<<std::endl;
}

#ifdef CPU_ONLY
STUB_GPU(MaskTransformerLayer);
#endif

INSTANTIATE_CLASS(MaskTransformerLayer);
REGISTER_LAYER_CLASS(MaskTransformer);

}  // namespace caffe

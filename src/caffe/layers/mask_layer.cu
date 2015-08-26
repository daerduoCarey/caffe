#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/mask_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaskTransformerForwardGPU(const int nthreads, const int C, 
		const int H, const int W, const Dtype* U, const Dtype* mm, Dtype* V) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		
		int i = index / (C*H*W);
		int s = (index / W) % H;
		int t = index % W;

		int mm_idx = i * (H*W) + s * W + t;

		if(mm[mm_idx] >= 0 && mm[mm_idx] <= 1) {
			V[index] = U[index] * mm[mm_idx];
		} else if(mm[mm_idx] > 1) {
			V[index] = U[index];
		}
	}	
}

template <typename Dtype>
void MaskTransformerLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	string prefix = "MaskTransformerLayer::Forward_gpu::\t";

	const Dtype* U = bottom[0]->gpu_data();
	const Dtype* mm = bottom[1]->gpu_data();
	
	Dtype* V = top[0]->mutable_gpu_data();

	caffe_gpu_set(top[0]->count(), (Dtype)0, V);

	const int nthreads = N * C * H * W;

	MaskTransformerForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	      CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, H, W, U, mm, V);
}

template <typename Dtype>
__global__ void MaskTransformerBackwardGPU_dmm(const int nthreads, const int C, 
		const int H, const int W, const Dtype* U, const Dtype* mm, 
		const Dtype* dV, Dtype* dmm) {

	CUDA_KERNEL_LOOP(index, nthreads) {
	
		int i = index / (H*W);
		int s = (index / W) % H;
		int t = index % W;
		
		int U_idx = i * (C*H*W) + s * W + t;
		if(mm[index] >= 0 && mm[index] <= 1) {
			dmm[index] += U[U_idx + 0 * (H*W)] * dV[U_idx + 0 * (H*W)];
			dmm[index] += U[U_idx + 1 * (H*W)] * dV[U_idx + 1 * (H*W)];
			dmm[index] += U[U_idx + 2 * (H*W)] * dV[U_idx + 2 * (H*W)];
		}
	}
}

template <typename Dtype>
__global__ void MaskTransformerBackwardGPU_dU(const int nthreads, const int C, 
		const int H, const int W, const Dtype* U, const Dtype* mm, 
		const Dtype* dV, Dtype* dU) {

	CUDA_KERNEL_LOOP(index, nthreads) {
	
		int i = index / (C*H*W);
		int s = (index / W) % H;
		int t = index % W;

		int mm_idx = i * (H*W) + s * W + t;

		if(mm[mm_idx] >= 0 && mm[mm_idx] <= 1) {
			dU[index] = mm[mm_idx] * dV[index];
		} else if(mm[mm_idx] > 1) {
			dU[index] = dV[index];
		}
	}
}

template <typename Dtype>
void MaskTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	string prefix = "MaskTransformerLayer::Backward_GPU::\t";

	const Dtype* dV = top[0]->gpu_diff();
	const Dtype* U = bottom[0]->gpu_data();
	const Dtype* mm = bottom[1]->gpu_data();

	Dtype* dmm = bottom[1]->mutable_gpu_diff();
	caffe_gpu_set(bottom[1]->count(), (Dtype)0., dmm);

	const int nthreads = N * H * W;

	MaskTransformerBackwardGPU_dmm<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, H, W, U, mm, dV, dmm);

	if(to_compute_dU_) {
		Dtype* dU = bottom[0]->mutable_gpu_diff();
		caffe_gpu_set(bottom[0]->count(), (Dtype)0., dU);
		const int nthreads = N * C * H * W;
		MaskTransformerBackwardGPU_dU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, H, W, U, mm, dV, dU);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(MaskTransformerLayer);

}	// namespace caffe

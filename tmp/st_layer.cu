#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/st_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SpatialTransformerForwardGPU(const int nthreads, int N, int C,
		int output_H_, int output_W_, int H, int W,
		Dtype* input_grid_data, const Dtype* U, Dtype* V) {
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int j = (index / (output_W_ * output_H_)) % C;
		const int i = index / (output_W_ * output_H_ * C);

		Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
		const int row_idx = output_W_ * s + t;

	  	const Dtype px = coordinates[row_idx * 2];
	  	const Dtype py = coordinates[row_idx * 2 + 1];

	  	const int V_offset = i * (C * output_H_ * output_W_) + j * (output_H_ * output_W_)
	  			+ s * output_W_ + t;

	  	V[V_offset] = (Dtype)0.;

	  	const Dtype x = (px + 1) / 2 * H;
	  	const Dtype y = (py + 1) / 2 * W;

	  	int m, n; Dtype w;
	  	const Dtype* pic = U + i * (C * H * W) + j * (H * W);

	  	m = floor(x); n = floor(y); w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = fmaxf(0, 1 - abs(x - m)) * fmaxf(0, 1 - abs(y - n));
	  		V[V_offset] += w * pic[m * W + n];
	  	}

	  	m = floor(x) + 1; n = floor(y); w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = fmaxf(0, 1 - abs(x - m)) * fmaxf(0, 1 - abs(y - n));
	  		V[V_offset] += w * pic[m * W + n];
	  	}

	  	m = floor(x); n = floor(y) + 1; w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = fmaxf(0, 1 - abs(x - m)) * fmaxf(0, 1 - abs(y - n));
	  		V[V_offset] += w * pic[m * W + n];
	  	}

	  	m = floor(x) + 1; n = floor(y) + 1; w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = fmaxf(0, 1 - abs(x - m)) * fmaxf(0, 1 - abs(y - n));
	  		V[V_offset] += w * pic[m * W + n];
	  	}
  }
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	const Dtype* U = bottom[0]->gpu_data();
	const Dtype* theta = bottom[1]->gpu_data();
	const Dtype* output_grid_data = output_grid->gpu_data();

	Dtype* input_grid_data = input_grid->mutable_gpu_data();
	Dtype* V = top[0]->mutable_gpu_data();

	caffe_gpu_set(input_grid->count(), (Dtype)0, input_grid_data);
	caffe_gpu_set(top[0]->count(), (Dtype)0, V);

	N = bottom[0]->shape(0);
	C = bottom[0]->shape(1);
	H = bottom[0]->shape(2);
	W = bottom[0]->shape(3);

	// compute out input_grid_data
	for(int i = 0; i < N; ++i) {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, output_H_ * output_W_, 2, 3, (Dtype)1.,
				output_grid_data, theta + 6 * i, (Dtype)0.,
				input_grid_data + (output_H_ * output_W_ * 2) * i);
	}

	const int nthreads = N * C * output_H_ * output_W_;

	SpatialTransformerForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N, C, output_H_, output_W_, H, W, input_grid_data, U, V);

}

template <typename Dtype>
__global__ void SpatialTransformerBackwardGPU(const int nthreads, int C,
		int output_H_, int output_W_, int H, int W,
		const Dtype* input_grid_data, const Dtype* dV_array, Dtype* input_grid_diff,
		Dtype* dTheta, Dtype* dU, const Dtype* U_array) {
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int i = index;

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;
		Dtype* coordinates_diff = input_grid_diff + (output_H_ * output_W_ * 2) * i;

		int row_idx; Dtype px, py, dpx, dpy, delta_dpx, delta_dpy;

		for(int s = 0; s < output_H_; ++s)
			for(int t = 0; t < output_W_; ++t) {

				row_idx = output_W_ * s + t;

				px = coordinates[row_idx * 2];
				py = coordinates[row_idx * 2 + 1];

				for(int j = 0; j < C; ++j) {

					delta_dpx = delta_dpy = (Dtype)0.;

					const Dtype x = (px + 1) / 2 * H;
					const Dtype y = (py + 1) / 2 * W;
					const int dV_offset = i * (C * output_H_ * output_W_) + j * (output_H_ * output_W_)
							+ s * output_W_ + t;
					const Dtype dV = dV_array[dV_offset];

					int m, n; Dtype w;
					const Dtype* U = U_array + i * (C * H * W) + j * (H * W);

					m = floor(x); n = floor(y); w = 0;
					if(m >= 0 && m < H && n >= 0 && n < W) {
						w = fmaxf(0, 1 - abs(x - m)) * fmaxf(0, 1 - abs(y - n));

						int tmp_offset = i * (C * H * W) + j * (H * W);
						dU[tmp_offset + m * W + n] += w * dV;

						if(abs(x - m) < 1) {
							if(m >= x) {
								delta_dpx += fmaxf(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
							} else {
								delta_dpx -= fmaxf(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
							}
						}

						if(abs(y - n) < 1) {
							if(n >= y) {
								delta_dpy += fmaxf(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
							} else {
								delta_dpy -= fmaxf(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
							}
						}
					}

					m = floor(x) + 1; n = floor(y); w = 0;
					if(m >= 0 && m < H && n >= 0 && n < W) {
						w = fmaxf(0, 1 - abs(x - m)) * fmaxf(0, 1 - abs(y - n));

						int tmp_offset = i * (C * H * W) + j * (H * W);
						dU[tmp_offset + m * W + n] += w * dV;

						if(abs(x - m) < 1) {
							if(m >= x) {
								delta_dpx += fmaxf(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
							} else {
								delta_dpx -= fmaxf(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
							}
						}

						if(abs(y - n) < 1) {
							if(n >= y) {
								delta_dpy += fmaxf(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
							} else {
								delta_dpy -= fmaxf(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
							}
						}
					}

					m = floor(x); n = floor(y) + 1; w = 0;
					if(m >= 0 && m < H && n >= 0 && n < W) {
						w = fmaxf(0, 1 - abs(x - m)) * fmaxf(0, 1 - abs(y - n));

						int tmp_offset = i * (C * H * W) + j * (H * W);
						dU[tmp_offset + m * W + n] += w * dV;

						if(abs(x - m) < 1) {
							if(m >= x) {
								delta_dpx += fmaxf(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;

							} else {
								delta_dpx -= fmaxf(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
							}
						}

						if(abs(y - n) < 1) {
							if(n >= y) {
								delta_dpy += fmaxf(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
							} else {
								delta_dpy -= fmaxf(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
							}
						}
					}

					m = floor(x) + 1; n = floor(y) + 1; w = 0;
					if(m >= 0 && m < H && n >= 0 && n < W) {
						w = fmaxf(0, 1 - abs(x - m)) * fmaxf(0, 1 - abs(y - n));

						int tmp_offset = i * (C * H * W) + j * (H * W);
						dU[tmp_offset + m * W + n] += w * dV;

						if(abs(x - m) < 1) {
							if(m >= x) {
								delta_dpx += fmaxf(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
							} else {
								delta_dpx -= fmaxf(0, 1 - abs(y - n)) * U[m * W + n] * dV * H / 2;
							}
						}

						if(abs(y - n) < 1) {
							if(n >= y) {
								delta_dpy += fmaxf(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
							} else {
								delta_dpy -= fmaxf(0, 1 - abs(x - m)) * U[m * W + n] * dV * W / 2;
							}
						}
					}

					coordinates_diff[row_idx * 2] += delta_dpx;
					coordinates_diff[row_idx * 2 + 1] += delta_dpy;
				}

				dpx = coordinates_diff[row_idx * 2];
				dpy = coordinates_diff[row_idx * 2 + 1];

				dTheta[6 * i] += dpx * (s * 1.0 / output_H_ * 2 - 1);
				dTheta[6 * i + 1] += dpx * (t * 1.0 / output_W_ * 2 - 1);
				dTheta[6 * i + 2] += dpx;
				dTheta[6 * i + 3] += dpy * (s * 1.0 / output_H_ * 2 - 1);
				dTheta[6 * i + 4] += dpy * (t * 1.0 / output_W_ * 2 - 1);
				dTheta[6 * i + 5] += dpy;
			}
	}
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype* dV = top[0]->gpu_diff();
	const Dtype* input_grid_data = input_grid->gpu_data();
	const Dtype* U = bottom[0]->gpu_data();

	Dtype* dU = bottom[0]->mutable_gpu_diff();
	Dtype* dTheta = bottom[1]->mutable_gpu_diff();
	Dtype* input_grid_diff = input_grid->mutable_gpu_diff();

	caffe_gpu_set(bottom[0]->count(), (Dtype)0, dU);
	caffe_gpu_set(bottom[1]->count(), (Dtype)0, dTheta);
	caffe_gpu_set(input_grid->count(), (Dtype)0, input_grid_diff);

	const int nthreads = N;

	SpatialTransformerBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, output_H_, output_W_, H, W, input_grid_data,
					dV, input_grid_diff, dTheta, dU, U);

}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialTransformerLayer);

}	// namespace caffe

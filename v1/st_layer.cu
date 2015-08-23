#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/st_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SpatialTransformerForwardGPU(const int nthreads, int N, int C,
		int output_H_, int output_W_, int H, int W, const Dtype* U, 
		const Dtype* theta, Dtype* V, Dtype* input_grid_data) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int t = index % output_W_;
		const int s = (index / output_W_) % output_H_;
		const int j = (index / (output_W_ * output_H_)) % C;
		const int i = index / (output_W_ * output_H_ * C);

	  	const double output_x = s * 2.0 / output_H_ - 1;
		const double output_y = t * 2.0 / output_W_ - 1;
		
		const int theta_offset = i * 6;
		const double px = theta[theta_offset] * output_x + theta[theta_offset + 1] * output_y + theta[theta_offset + 2];
		const double py = theta[theta_offset + 3] * output_x + theta[theta_offset + 4] * output_y + theta[theta_offset + 5];

		const int input_grid_offset = i * (output_H_ * output_W_) + s * output_W_ + t;
		input_grid_data[input_grid_offset * 2] = px;
		input_grid_data[input_grid_offset * 2 + 1] = py;

	  	const int V_offset = i * (C * output_H_ * output_W_) + j * (output_H_ * output_W_)
	  			+ s * output_W_ + t;

	  	V[V_offset] = (Dtype)0.;

	  	const Dtype x = (px + 1) / 2 * H;
	  	const Dtype y = (py + 1) / 2 * W;

	  	int m, n; Dtype w;
	  	const Dtype* pic = U + i * (C * H * W) + j * (H * W);

	  	m = floor(x); n = floor(y); w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (x - m)) * (1 - (y - n));
	  		V[V_offset] += w * pic[m * W + n];
	  	}

	  	m = floor(x) + 1; n = floor(y); w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (m - x)) * (1 - (y - n));
	  		V[V_offset] += w * pic[m * W + n];
	  	}

	  	m = floor(x); n = floor(y) + 1; w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (x - m)) * (1 - (n - y));
	  		V[V_offset] += w * pic[m * W + n];
	  	}

	  	m = floor(x) + 1; n = floor(y) + 1; w = 0;
	  	if(m >= 0 && m < H && n >= 0 && n < W) {
	  		w = (1 - (m - x)) * (1 - (n - y));
	  		V[V_offset] += w * pic[m * W + n];
	  	}
  }
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	string prefix = "SpatialTransformerLayer::Forward_gpu::\t";
	
	const Dtype* U = bottom[0]->gpu_data();
	const Dtype* theta = bottom[1]->gpu_data();

	Dtype* V = top[0]->mutable_gpu_data();
	Dtype* input_grid_data = input_grid->mutable_gpu_data();

	caffe_gpu_set(top[0]->count(), (Dtype)0, V);

	const int nthreads = N * C * output_H_ * output_W_;
	
	SpatialTransformerForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
	      CAFFE_CUDA_NUM_THREADS>>>(nthreads, N, C, output_H_, output_W_, 
			      H, W, U, theta, V, input_grid_data);
}

template <typename Dtype>
__global__ void SpatialTransformerBackwardGPU(const int nthreads, int C, 
		int output_H_, int output_W_, int H, int W, const Dtype* input_grid_data, 
		const Dtype* dV_array, Dtype* dTheta, Dtype* dU, const Dtype* U_array) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int i = index;

		const Dtype* coordinates = input_grid_data + (output_H_ * output_W_ * 2) * i;

		int row_idx; Dtype px, py, dpx, dpy;

		for(int s = 0; s < output_H_; ++s)
			for(int t = 0; t < output_W_; ++t) {

				row_idx = output_W_ * s + t;

				px = coordinates[row_idx * 2];
				py = coordinates[row_idx * 2 + 1];

				dpx = dpy = (Dtype)0.;
				
				const Dtype x = (px + 1) / 2 * H;
				const Dtype y = (py + 1) / 2 * W;
				
				for(int j = 0; j < C; ++j) {

					const int dV_offset = i * (C * output_H_ * output_W_) + j * (output_H_ * output_W_)
							+ s * output_W_ + t;
					const Dtype dV = dV_array[dV_offset];

					int m, n; Dtype w;
					const Dtype* U = U_array + i * (C * H * W) + j * (H * W);

					m = floor(x); n = floor(y); w = 0;
					if(m >= 0 && m < H && n >= 0 && n < W) {
						w = (1 - (x - m)) * (1 - (y - n));

						int tmp_offset = i * (C * H * W) + j * (H * W);
						dU[tmp_offset + m * W + n] += w * dV;

						dpx -= (1 - (y - n)) * U[m * W + n] * dV * H / 2;
						dpy -= (1 - (x - m)) * U[m * W + n] * dV * W / 2;
					}

					m = floor(x) + 1; n = floor(y); w = 0;
					if(m >= 0 && m < H && n >= 0 && n < W) {
						w = (1 - (m - x)) * (1 - (y - n));

						int tmp_offset = i * (C * H * W) + j * (H * W);
						dU[tmp_offset + m * W + n] += w * dV;

						dpx += (1 - (y - n)) * U[m * W + n] * dV * H / 2;
						dpy -= (1 - (m - x)) * U[m * W + n] * dV * W / 2;
					}

					m = floor(x); n = floor(y) + 1; w = 0;
					if(m >= 0 && m < H && n >= 0 && n < W) {
						w = (1 - (x - m)) * (1 - (n - y));

						int tmp_offset = i * (C * H * W) + j * (H * W);
						dU[tmp_offset + m * W + n] += w * dV;

						dpx -= (1 - (n - y)) * U[m * W + n] * dV * H / 2;
						dpy += (1 - (x - m)) * U[m * W + n] * dV * W / 2;
					}

					m = floor(x) + 1; n = floor(y) + 1; w = 0;
					if(m >= 0 && m < H && n >= 0 && n < W) {
						w = (1 - (m - x)) * (1 - (n - y));

						int tmp_offset = i * (C * H * W) + j * (H * W);
						dU[tmp_offset + m * W + n] += w * dV;

						dpx += (1 - (n - y)) * U[m * W + n] * dV * H / 2;
						dpy += (1 - (m - x)) * U[m * W + n] * dV * W / 2;
					}
				}

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

	string prefix = "SpatialTransformerLayer::Backward_gpu::\t";

	const Dtype* dV = top[0]->gpu_diff();
	const Dtype* U = bottom[0]->gpu_data();
	const Dtype* input_grid_data = input_grid->gpu_data();

	Dtype* dU = bottom[0]->mutable_gpu_diff();
	Dtype* dTheta = bottom[1]->mutable_gpu_diff();

	caffe_gpu_set(bottom[0]->count(), (Dtype)0, dU);
	caffe_gpu_set(bottom[1]->count(), (Dtype)0, dTheta);

	const int nthreads = N;

	SpatialTransformerBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
			CAFFE_CUDA_NUM_THREADS>>>(nthreads, C, output_H_, output_W_, H, W, input_grid_data,
					dV, dTheta, dU, U);
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialTransformerLayer);

}	// namespace caffe

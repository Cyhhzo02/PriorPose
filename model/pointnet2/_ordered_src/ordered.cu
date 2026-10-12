#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

template<bool Interpolate>
__global__ void ordered_kernel(const float* grad, const int64_t* positions,
    const int64_t* offsets, const float* weights, float* output,
    int64_t batch, int64_t channels, int64_t length, int64_t count) {
  const int64_t total = batch * channels * count;
  for (int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       tid < total; tid += (int64_t)blockDim.x * gridDim.x) {
    const int64_t destination = tid % count;
    const int64_t bc = tid / count;
    const int64_t b = bc / channels;
    const int64_t begin = offsets[b * (count + 1) + destination];
    const int64_t end = offsets[b * (count + 1) + destination + 1];
    const int64_t source_length = Interpolate ? length / 3 : length;
    float sum = 0.0f;
    for (int64_t j = begin; j < end; ++j) {
      const int64_t source = positions[b * length + j];
      float value;
      if (Interpolate) {
        value = __fmul_rn(grad[bc * source_length + source / 3],
                          weights[b * length + source]);
      } else {
        value = grad[bc * source_length + source];
      }
      // No atomic, no tree reordering, no fused multiply-add.
      sum = __fadd_rn(sum, value);
    }
    output[tid] = sum;
  }
}

at::Tensor ordered_reduce_cuda(at::Tensor grad, at::Tensor positions,
    at::Tensor offsets, at::Tensor weight, int64_t count, bool interpolation) {
  TORCH_CHECK(grad.is_cuda() && positions.is_cuda() && offsets.is_cuda() && weight.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(grad.device()==positions.device() && grad.device()==offsets.device() && grad.device()==weight.device(), "Device mismatch");
  TORCH_CHECK(grad.scalar_type()==at::kFloat && weight.scalar_type()==at::kFloat, "FP32 only");
  TORCH_CHECK(positions.scalar_type()==at::kLong && offsets.scalar_type()==at::kLong, "Int64 index tensors required");
  TORCH_CHECK(grad.is_contiguous() && positions.is_contiguous() && offsets.is_contiguous() && weight.is_contiguous(), "Contiguous tensors required");
  TORCH_CHECK(grad.dim()==3 && positions.dim()==2 && offsets.dim()==2 && count>0, "Invalid shapes/count");
  const auto batch=grad.size(0), channels=grad.size(1), length=positions.size(1);
  TORCH_CHECK(positions.size(0)==batch && offsets.size(0)==batch && offsets.size(1)==count+1, "Shape mismatch");
  TORCH_CHECK(grad.size(2)*(interpolation ? 3 : 1)==length, "Gradient length mismatch");
  TORCH_CHECK(!interpolation || weight.numel()==batch*length, "Weight size mismatch");
  const c10::cuda::CUDAGuard guard(grad.device());
  auto output=at::empty({batch,channels,count},grad.options());
  const int64_t total=batch*channels*count;
  if (total==0) return output;
  const int blocks=(int)std::min<int64_t>((total+255)/256,65535);
  const auto stream=c10::cuda::getCurrentCUDAStream().stream();
  if (interpolation) {
    ordered_kernel<true><<<blocks,256,0,stream>>>(grad.data_ptr<float>(),positions.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),weight.data_ptr<float>(),output.data_ptr<float>(),batch,channels,length,count);
  } else {
    ordered_kernel<false><<<blocks,256,0,stream>>>(grad.data_ptr<float>(),positions.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(),weight.data_ptr<float>(),output.data_ptr<float>(),batch,channels,length,count);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

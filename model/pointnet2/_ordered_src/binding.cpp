#include <torch/extension.h>
at::Tensor ordered_reduce_cuda(at::Tensor grad, at::Tensor positions,
    at::Tensor offsets, at::Tensor weight, int64_t count, bool interpolation);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("reduce", &ordered_reduce_cuda, "Fixed serial order per destination, GPU parallel destinations");
}

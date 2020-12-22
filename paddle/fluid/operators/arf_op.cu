/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/prelu.h"
#include "paddle/fluid/operators/prelu_op.h"
#include "paddle/fluid/operators/reduce_ops/cub_reduce.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

#define CUDA_NUM_THREADS 1024

inline static int PADDLE_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename DeviceContext, typename T>
class CUDAARFKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<Tensor>("Input");
    auto* dinput = context.Output<Tensor>(framework::GradVarName("Input"));
    auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* indices = context.Input<Tensor>("Indices");


    const T* input_ptr = input->data<T>();
    const T* dout_ptr = dout->data<T>();


    // input size
    auto input_dims = input->dims();;
    int nOutputPlane = input_dims[0];
    int nInputPlane = input_dims[1];
    int nOrientation = input_dims[2];
    int kH = input_dims[4];
    int kW = input_dims[5];
    nOutputPlane = nOutputPlane;
    nInputPlane = nInputPlane;
    nOrientation = nOrientation;
    kH = kH;
    kW = kW;

    VLOG(4) << "nOutputPlane:" << nOutputPlane << ", nInputPlane:" << nInputPlane
            << "nOrientation:" << nOrientation << "kH" << kH << "kW" << kW;

  }
};


template <typename T>
__global__ void ARFOpGradKernel(const T* x_ptr, const T* alpha_ptr,
                                  const T* dy_ptr, T* dx_ptr, T* dalpha_ptr,
                                  size_t channel_num, size_t plane_size,
                                  size_t spatial_size, size_t numel,
                                  PRELU_MODE mode) {
  CUDA_KERNEL_LOOP(index, numel) {
    T scale;

    T x = x_ptr[index];
    T dy = dy_ptr[index];
    if (dx_ptr != nullptr) dx_ptr[index] = (x > 0) ? dy : scale * dy;
    if (dalpha_ptr != nullptr) dalpha_ptr[index] = (x > 0) ? 0 : x * dy;
  }
}

template <typename T>
class ARFOpGradFunctor {
 public:
  void operator()(cudaStream_t stream, const T* x, const T* alpha, const T* dy,
                  T* dx, T* dalpha, const framework::DDim& input_dims,
                  PRELU_MODE mode) {
    size_t numel = 1;
    for (size_t i = 0; i < input_dims.size(); ++i) {
      numel *= input_dims[i];
    }
    size_t plane_size = numel / input_dims[0] / input_dims[1];
    size_t spatial_size = numel / input_dims[0];

    PReluOpGradKernel<
        T><<<PADDLE_GET_BLOCKS(numel), CUDA_NUM_THREADS, 0, stream>>>(
        x, alpha, dy, dx, dalpha, input_dims[1], plane_size, spatial_size,
        numel, mode);
  }
};

template <typename T>
struct IdentityFunctor {
  HOSTDEVICE inline T operator()(const T& x) const { return x; }
};

template <typename DeviceContext, typename T>
class CUDAARFGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* alpha = context.Input<Tensor>("Alpha");
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dalpha = context.Output<Tensor>(framework::GradVarName("Alpha"));

    const T* x_ptr = x->data<T>();
    const T* alpha_ptr = alpha->data<T>();
    const T* dy_ptr = dy->data<T>();
    T* dx_ptr = dx ? dx->mutable_data<T>(context.GetPlace()) : nullptr;
    T* dalpha_ptr =
        dalpha ? dalpha->mutable_data<T>(context.GetPlace()) : nullptr;

    if (!dx && !dalpha) return;

    auto& mode = context.Attr<std::string>("mode");

    int numel = x->numel();
    auto dim = x->dims();
    std::vector<int> input_shape = framework::vectorize<int>(dim);
    auto stream = context.cuda_device_context().stream();

    T* dalpha_tmp_ptr;
    Tensor dalpha_tmp;
    if (dalpha_ptr == nullptr) {
      dalpha_tmp_ptr = dalpha_ptr;
    } else {
      auto& dev_ctx = context.template device_context<DeviceContext>();
      dalpha_tmp = context.AllocateTmpTensor<T, DeviceContext>(dim, dev_ctx);
      dalpha_tmp_ptr = dalpha_tmp.mutable_data<T>(context.GetPlace());
    }

    PRELU_MODE m;
    if (mode == "element") {
      m = Element;
    } else if (mode == "channel") {
      m = Channel;
    } else {
      m = Scalar;
    }
    PreluOpGradFunctor<T> prelu_grad;
    prelu_grad(stream, x_ptr, alpha_ptr, dy_ptr, dx_ptr, dalpha_tmp_ptr, dim,
               m);

    if (dalpha_tmp_ptr == nullptr) return;

    std::vector<int> reduce_dims;
    for (size_t i = 0; i < dim.size(); i++) {
      if (mode == "channel" && i == 1) continue;
      if (mode == "element" && i != 0) continue;
      reduce_dims.push_back(i);
    }

    TensorReduce<T, T, cub::Sum, IdentityFunctor<T>>(
        dalpha_tmp, dalpha, reduce_dims, static_cast<T>(0), cub::Sum(),
        IdentityFunctor<T>(), stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    arf, ops::CUDAARFKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CUDAARFKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    arf_grad,
    ops::CUDAARFGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CUDAARFGradKernel<paddle::platform::CUDADeviceContext, double>);

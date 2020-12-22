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

#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/layout_utils.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
constexpr int kConvMKLDNNFP32 = 1;
constexpr int kConvMKLDNNINT8 = 2;
constexpr int MaxKeyLength = 256;



// Define Op classes in .h file so that other conv
// operator implementations can reuse the code.
class ARFOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final;

 protected:
  virtual void Apply() {}
};


class ARFOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{
        {"Input", /*->*/ "Output"}};
    return m;
  }
};


class ARFOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    std::vector<int64_t> output_shape = ComputeOutputShape(ctx);

    OP_INOUT_CHECK(ctx->HasOutput("Output"), "Output", "Output", "Conv");
    ctx->SetOutputDim("Output", framework::make_ddim(output_shape));
    ctx->ShareLoD("Input", "Output");
  }

 protected:
  std::vector<int64_t> ComputeOutputShape(
      framework::InferShapeContext* ctx) const;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override;
};

class ARFOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override;
};


template <typename DeviceContext, typename T>
class CPUARFKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<Tensor>("Input");
    auto* indices = context.Input<Tensor>("Indices");
    auto* output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());

    const std::string data_format = context.Attr<std::string>("data_format");
    //const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");


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


    // nOrientation, kH, kW, nRotation
    //auto indices_dims = transformed_indices.dims();
    //int nOrientation = indices_dims[0];
    //int kH = indices_dims[1];
    //int kW = indices_dims[2];
    //int nRotation = indices_dims[3];
    //nRotation = nRotation;
    auto indices_dims = indices->dims();
    //int nOrientation = indices_dims[0];
    //int kH = indices_dims[1];
    //int kW = indices_dims[2];
    int nRotation = indices_dims[3];
    nRotation = nRotation;
    //printf('%d %d %d %d %d %d', nOutputPlane, nInputPlane, nOrientation, kH, kW, nRotation)


    //Tensor transformed_input(input->type());
    //Tensor transformed_output(output->type());

    //transformed_input = *input;
    //transformed_output = *output;
    // TODO: add ARF CPU kernel

    const T* input_ptr = input->data<T>();
    const T* indices_ptr = indices->data<T>();
    T* o_ptr = output->mutable_data<T>(context.GetPlace());

    //input_ptr = input_ptr;

    int i, j, l;
    int k;
    const int nEntry = nOrientation * kH * kW;
    //#pragma omp parallel for private(i, j, l, k)
      for (i = 0; i < nOutputPlane; i++) {
        for (j = 0; j < nInputPlane; j++) {
          for (l = 0; l < nEntry; l++) {
            int weightIndex = i * nInputPlane * nEntry
                                + j * nEntry
                                + l;
            T val = *(input_ptr + weightIndex);
            // T val = *(weightData++);
            for (k = 0; k < nRotation; k++) {
              //int index = (uint16)(*(indicesData + l * nRotation + k)) - 1;
              unsigned short int index = (unsigned short int)(*(indices_ptr + l * nRotation + k)) - 1;
              T *target = o_ptr + i * (nRotation * nInputPlane * nEntry)
                                    + k * (nInputPlane * nEntry)
                                    + j * (nEntry)
                                    + index;
              *target = val;
            }
          }
        }
      }
    }

};



template <typename DeviceContext, typename T>
class CPUARFGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<Tensor>("Input");
    auto* dinput = context.Output<Tensor>(framework::GradVarName("Input"));
    auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* indices = context.Input<Tensor>("Indices");


    const T* input_ptr = input->data<T>();
    const T* dinput_ptr = dinput->data<T>();
    const T* indices_ptr = indices->data<T>();
    const T* dout_ptr = dout->data<T>();
    input_ptr = input_ptr;
    dinput_ptr = dinput_ptr;
    indices_ptr = indices_ptr;
    dout_ptr = dout_ptr;

    // input size
    input = input;
    input_ptr = input_ptr;
    //auto input_dims = input->dims();;

    // indices
    auto indices_dims = indices->dims();
    int nOrientation = indices_dims[0];
    int kH = indices_dims[1];
    int kW = indices_dims[2];
    int nRotation = indices_dims[3];
    
    // dout
    auto dout_dims = dout->dims();
    const int nOutputPlane = dout_dims[0] / nRotation;
    const int nInputPlane = dout_dims[0] / nOrientation;
    
    int nEntry = nOrientation * kH * kW;

    int i, j, l, k;
    if (dinput)
    {
        T* dinput_ptr = dinput->mutable_data<T>(context.GetPlace());

        for (i = 0; i < nOutputPlane; i++) {
            for (j = 0; j < nInputPlane; j++) {
                for (l = 0; l < nEntry; l++) {
                    int gradInputIndex = i * nInputPlane * nEntry
                    + j * nEntry
                    + l;
                    T *val = dinput_ptr + gradInputIndex;
                    // T *val = gradInputData++;
                    *val = 0;
                    for (k = 0; k < nRotation; k++) {
                    unsigned short int index = (unsigned short int)(*(indices_ptr + l * nRotation + k)) - 1;
                    const T *target = dout_ptr + i * (nRotation * nInputPlane * nEntry)
                    + k * (nInputPlane * nEntry)
                    + j * (nEntry)
                    + index;
                    *val = *val + *target;
                    }
                }
            }
        }
    }

  }

};



}  // namespace operators
}  // namespace paddle

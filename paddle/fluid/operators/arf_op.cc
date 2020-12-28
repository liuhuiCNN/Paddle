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

#include "paddle/fluid/operators/arf_op.h"

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_version_registry.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/operators/conv_cudnn_op_cache.h"
#include "paddle/fluid/platform/cudnn_helper.h"
#endif
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
#include "paddle/fluid/platform/cudnn_workspace_helper.h"

namespace paddle {
namespace operators {


std::vector<int64_t> ARFOp::ComputeOutputShape(
    framework::InferShapeContext* ctx) const {
  // check
  OP_INOUT_CHECK(ctx->HasInput("InputWeight"), "Input", "InputWeight", "ARF");

  auto inputweigt_dims = ctx->GetInputDim("InputWeight");
  auto indices_dims = ctx->GetInputDim("Indices");
  const std::string data_format = ctx->Attrs().Get<std::string>("data_format");

  // MKL-DNN Kernels are using NCHW order of dims description
  // so we ignore data_format consideration for MKL-DNN kernel
  //const bool channel_last = (this->IsMKLDNNType() == false) &&
  //                          (data_format == "NHWC" || data_format == "NDHWC");

  PADDLE_ENFORCE_EQ(
      inputweigt_dims.size() == 5, true,
      platform::errors::InvalidArgument(
          "The input of Op(ARF) should be a 5-D Tensor. But "
          "received: input's dimension is %u, input's shape is [%s].",
          inputweigt_dims.size(), inputweigt_dims));


  int nOutputPlane = inputweigt_dims[0];
  int nInputPlane = inputweigt_dims[1];
  int nOrientation = inputweigt_dims[2];
  int kH = inputweigt_dims[3];
  int kW = inputweigt_dims[4];


  int indices_nOrientation = indices_dims[0];
  int indices_kH = indices_dims[1];
  int indices_kW = indices_dims[2];
  int indices_nRotation = indices_dims[3];
  indices_nOrientation = indices_nOrientation;

  PADDLE_ENFORCE_EQ(
      kH, indices_kH,
      platform::errors::InvalidArgument(
          "kH should == indices_kH"));

  PADDLE_ENFORCE_EQ(
      kW, indices_kW,
      platform::errors::InvalidArgument(
          "kW should == indices_kW"));

  PADDLE_ENFORCE_EQ(
      nOrientation, indices_nOrientation,
      platform::errors::InvalidArgument(
          "nOrientation should == indices_nOrientation"));

  // output nOutputPlane * nRotation, nInputPlane * nOrientation, kH, kW
  std::vector<int64_t> output_shape({nOutputPlane * indices_nRotation});
  output_shape.push_back(nInputPlane * nOrientation);
  output_shape.push_back(kH);
  output_shape.push_back(kW);
  return output_shape;
}


framework::OpKernelType ARFOp::GetExpectedKernelType(
  // TODO: update GetExpectedKernelType
    const framework::ExecutionContext& ctx) const {
  int customized_type_value = framework::OpKernelType::kDefaultCustomizedTypeValue;
  framework::LibraryType library{framework::LibraryType::kPlain};
  // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
  auto inputweight_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "InputWeight");
  std::string data_format = "AnyLayout";  // todo enable data layout when it's ready
  framework::DataLayout layout = framework::StringToDataLayout(data_format);

#ifdef PADDLE_WITH_CUDA
  if (platform::CanCUDNNBeUsed(ctx)) {
    library = framework::LibraryType::kCUDNN;
  }
#endif

  /*
  if (inputweight_data_type == framework::proto::VarType::FP16) {
    PADDLE_ENFORCE_EQ(library, framework::LibraryType::kCUDNN,
                      platform::errors::InvalidArgument(
                          "float16 can only be used when CUDNN is used"));
  }*/

  auto type = framework::OpKernelType(inputweight_data_type, ctx.GetPlace(), layout, library, customized_type_value);
  return type;
}


framework::OpKernelType ARFOp::GetKernelTypeForVar(
    const std::string& var_name, const Tensor& tensor,
    const framework::OpKernelType& expected_kernel_type) const {
    return framework::OpKernelType(expected_kernel_type.data_type_, tensor.place(), tensor.layout());
}


void ARFOpMaker::Make() {
  AddAttr<bool>("is_test",
                "(bool, default false) Set to true for inference only, false "
                "for training. Some layers may run faster when this is true.")
      .SetDefault(false);
  AddInput("InputWeight",
           "(Tensor) The input weight of arf. "
           "The format of input tensor is NCHW or NHWC, where N is batch size, "
           "Input shape [nOutputPlan, nInputPlan, nOrientation, kH, kW]");
  AddInput("Indices",
           "(Tensor) The Indices tensor of arf operator. "
           "The format of the filter tensor is [nOrientation, kH, kW, nRotation]");
  AddOutput("Output",
            "(Tensor) The output tensor of arf operator. "
            "It has same data fromat and data type as the Input."
             "output should be [nOutputPlane * nRotation, nInputPlane * nOrientation, kH, kW]");

  AddAttr<bool>(
      "use_cudnn",
      "(bool, default false) Only used in cudnn kernel, need install cudnn")
      .SetDefault(false);
  AddAttr<bool>("use_mkldnn",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<std::string>(
      "mkldnn_data_type",
      "(string, default \"float32\"). Data type of mkldnn kernel")
      .SetDefault("float32")
      .InEnum({"float32", "int8", "bfloat16"});
  AddAttr<std::string>(
      "data_format",
      "(string, default \"NCDHW\"). Data format")
      .SetDefault("NCDHW");
  
  AddComment(R"DOC(
Convolution Operator.

ARF

Example:
  Input:
       Input shape: $()$
       Filter shape: $()$
  Output:
       Output shape: $()$
  Where

)DOC");
  Apply();
}


framework::OpKernelType ARFOpGrad::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  int customized_type_value =
      framework::OpKernelType::kDefaultCustomizedTypeValue;
  framework::LibraryType library_{framework::LibraryType::kPlain};

  // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
  std::string data_format = "AnyLayout";
  framework::DataLayout layout_ = framework::StringToDataLayout(data_format);

#ifdef PADDLE_WITH_CUDA
  if (platform::CanCUDNNBeUsed(ctx)) {
    library_ = framework::LibraryType::kCUDNN;
  }
#endif

  auto type = framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "InputWeight"), ctx.GetPlace(),
      layout_, library_, customized_type_value);
  return type;
}


framework::OpKernelType ARFOpGrad::GetKernelTypeForVar(
    const std::string& var_name, const Tensor& tensor,
    const framework::OpKernelType& expected_kernel_type) const {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                 tensor.place(), tensor.layout());
}


void ARFOpGrad::InferShape(framework::InferShapeContext* ctx) const {
  //OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "arf");
  OP_INOUT_CHECK(ctx->HasInput("InputWeight"), "Input", "InputWeight", "arf");

  OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Output")), "Input", "Output@GRAD", "arf");


  auto input_weight_dims = ctx->GetInputDim("InputWeight");
  int nOutputPlane = input_weight_dims[0];
  int nInputPlane = input_weight_dims[1];
  int nOrientation = input_weight_dims[2];
  int kH = input_weight_dims[3];
  int kW = input_weight_dims[4];
  nOutputPlane = nOutputPlane;
  nInputPlane = nInputPlane;
  nOrientation = nOrientation;
  kH = kH;
  kW = kW;

  auto indices_dims = ctx->GetInputDim("Indices");
  int indices_nRotation = indices_dims[3];


  printf("debug input_weight_dims %d %d nOrientation %d ro %d\n", int(input_weight_dims[0]), int(input_weight_dims[1]), indices_nRotation, nOrientation);
  
  if (ctx->HasOutput(framework::GradVarName("InputWeight"))) {
    ctx->SetOutputDim(framework::GradVarName("InputWeight"), input_weight_dims);
  }
}


template <typename T>
class ARFGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("InputWeight", this->Input("InputWeight"));
    op->SetInput("Indices", this->Input("Indices"));
    // d_out
    //op->SetInput(framework::GradVarName("Output"), this->OutputGrad("Output"));
    

    //op->SetInput(framework::GradVarName("Output"), this->Input(framework::GradVarName("Output")));
    op->SetInput(framework::GradVarName("Output"), this->OutputGrad("Output"));
    // d_inputweight
    op->SetOutput(framework::GradVarName("InputWeight"), this->InputGrad("InputWeight"));
    op->SetAttrMap(this->Attrs());
  }
};


}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(arf, ops::ARFOp,
                  ops::ARFOpMaker,
                  ops::ARFOpInferVarType,
                  ops::ARFGradMaker<paddle::framework::OpDesc>,
                  ops::ARFGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(arf_grad, ops::ARFOpGrad);//,
                  //ops::ARFGradMaker<paddle::framework::OpDesc>,
                  //ops::ARFGradMaker<paddle::imperative::OpBase>);


REGISTER_OP_CPU_KERNEL(
    arf, ops::CPUARFKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CPUARFKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    arf_grad,
    ops::CPUARFGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CPUARFGradKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_VERSION(arf)
    .AddCheckpoint(
        R"ROC(
      Upgrade arf, add a new attribute [use_addto].
    )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "use_addto",
            "In order to support new feature (inplace addto strategy) for "
            "gradient accumulation.",
            false));


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
  // TODO: HasInput???
  OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "Conv");
  //OP_INOUT_CHECK(ctx->HasInput("Indices"), "Input", "Filter", "Conv");

  auto input_dims = ctx->GetInputDim("Input");
  auto indices_dims = ctx->GetInputDim("Indices");
  const std::string data_format = ctx->Attrs().Get<std::string>("data_format");

  // MKL-DNN Kernels are using NCHW order of dims description
  // so we ignore data_format consideration for MKL-DNN kernel
  //const bool channel_last = (this->IsMKLDNNType() == false) &&
  //                          (data_format == "NHWC" || data_format == "NDHWC");

  PADDLE_ENFORCE_EQ(
      input_dims.size() == 5, true,
      platform::errors::InvalidArgument(
          "The input of Op(ARF) should be a 5-D Tensor. But "
          "received: input's dimension is %u, input's shape is [%s].",
          input_dims.size(), input_dims));


  int nOutputPlane = input_dims[0];
  int nInputPlane = input_dims[1];
  int nOrientation = input_dims[2];
  int kH = input_dims[3];
  int kW = input_dims[4];


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
  int customized_type_value =
      framework::OpKernelType::kDefaultCustomizedTypeValue;
  framework::LibraryType library{framework::LibraryType::kPlain};
  // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
  auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Input");
  std::string data_format =
      "AnyLayout";  // todo enable data layout when it's ready
  framework::DataLayout layout = framework::StringToDataLayout(data_format);

#ifdef PADDLE_WITH_CUDA
  if (platform::CanCUDNNBeUsed(ctx)) {
    library = framework::LibraryType::kCUDNN;
  }
#endif
#ifdef PADDLE_WITH_MKLDNN
  if (library == framework::LibraryType::kPlain && this->CanMKLDNNBeUsed(ctx)) {
    library = framework::LibraryType::kMKLDNN;
    layout = framework::DataLayout::kMKLDNN;
    customized_type_value =
        (input_data_type == framework::DataTypeTrait<int8_t>::DataType() ||
         input_data_type == framework::DataTypeTrait<uint8_t>::DataType())
            ? kConvMKLDNNINT8
            : kConvMKLDNNFP32;
  }
#endif

  if (input_data_type != framework::proto::VarType::INT8 &&
      input_data_type != framework::proto::VarType::UINT8 &&
      input_data_type != framework::proto::VarType::BF16) {
    auto filter_data_type = ctx.Input<Tensor>("Filter")->type();
    PADDLE_ENFORCE_EQ(input_data_type, filter_data_type,
                      platform::errors::InvalidArgument(
                          "input and filter data type should be consistent"));
  }
  if (input_data_type == framework::proto::VarType::FP16) {
    PADDLE_ENFORCE_EQ(library, framework::LibraryType::kCUDNN,
                      platform::errors::InvalidArgument(
                          "float16 can only be used when CUDNN is used"));
  }

  auto type = framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                      library, customized_type_value);
  return type;
}

framework::OpKernelType ARFOp::GetKernelTypeForVar(
    const std::string& var_name, const Tensor& tensor,
    const framework::OpKernelType& expected_kernel_type) const {
#ifdef PADDLE_WITH_MKLDNN
  // Only input require reshaping, weights and
  // bias are having shape in NCHW order
  if ((var_name == "Input") &&
      (expected_kernel_type.data_layout_ == framework::DataLayout::kMKLDNN) &&
      (tensor.layout() != framework::DataLayout::kMKLDNN)) {
    auto attrs = Attrs();
    auto ar = paddle::framework::AttrReader(attrs);
    const std::string data_format = ar.Get<std::string>("data_format");
    auto dl = framework::StringToDataLayout(data_format);
    // Some models may have intentionally set "AnyLayout" for conv
    // op. Treat this as NCHW (default data_format value)
    if (dl != framework::DataLayout::kAnyLayout) {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), dl);
    }
  }
#endif
  return framework::OpKernelType(expected_kernel_type.data_type_,
                                 tensor.place(), tensor.layout());
}

void ARFOpMaker::Make() {
  AddAttr<bool>("is_test",
                "(bool, default false) Set to true for inference only, false "
                "for training. Some layers may run faster when this is true.")
      .SetDefault(false);
  AddInput("Input",
           "(Tensor) The input tensor of convolution operator. "
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
#ifdef PADDLE_WITH_MKLDNN
  if (library_ == framework::LibraryType::kPlain &&
      this->CanMKLDNNBeUsed(ctx)) {
    const std::string data_format = ctx.Attr<std::string>("data_format");
    library_ = framework::LibraryType::kMKLDNN;
    layout_ = framework::DataLayout::kMKLDNN;
    customized_type_value = kConvMKLDNNFP32;
  }
#endif

  auto type = framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "Input"), ctx.GetPlace(),
      layout_, library_, customized_type_value);
  return type;
}

framework::OpKernelType ARFOpGrad::GetKernelTypeForVar(
    const std::string& var_name, const Tensor& tensor,
    const framework::OpKernelType& expected_kernel_type) const {
#ifdef PADDLE_WITH_MKLDNN
  // Only input require reshaping, weights and
  // bias are having shape in NCHW order
  if (((var_name == "Input") ||
       (var_name == framework::GradVarName("Output"))) &&
      (expected_kernel_type.data_layout_ == framework::DataLayout::kMKLDNN) &&
      (tensor.layout() != framework::DataLayout::kMKLDNN)) {
    auto attrs = Attrs();
    auto ar = paddle::framework::AttrReader(attrs);
    const std::string data_format = ar.Get<std::string>("data_format");
    auto dl = framework::StringToDataLayout(data_format);
    // Some models may have intentionally set "AnyLayout" for pool
    // op. Treat this as NCHW (default data_format value)
    if (dl != framework::DataLayout::kAnyLayout) {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), dl);
    }
  }
#endif
  return framework::OpKernelType(expected_kernel_type.data_type_,
                                 tensor.place(), tensor.layout());
}


template <typename T>
class ARFGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("Input", this->Input("Input"));
    op->SetInput("Indices", this->Input("Indices"));
    op->SetInput(framework::GradVarName("Output"), this->OutputGrad("Output"));

    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetAttrMap(this->Attrs());
  }
};


}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(arf, ops::ARFOp, ops::ARFOpMaker,
                  ops::ARFOpInferVarType,
                  ops::ARFGradMaker<paddle::framework::OpDesc>,
                  ops::ARFGradMaker<paddle::imperative::OpBase>);

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


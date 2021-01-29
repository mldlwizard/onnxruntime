// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "TCS.h"
#include "core/providers/common.h"
#include "core/util/math_cpuonly.h"
#include <cmath>
#include <cfenv>

namespace onnxruntime {

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    TCS,
    11,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    TCS<float>);


// formula is Y = 2*A+B
template <typename T>
Status TCS<T>::Compute(OpKernelContext* ctx) const {

  const auto& A = *ctx->Input<Tensor>(0);
  const auto& B = *ctx->Input<Tensor>(1);
  
  auto& C = *ctx->Output(0, A.Shape());
  auto* input1 = A.template Data<T>();
  auto* input2 = B.template Data<T>();

  auto* output = C.template MutableData<T>();

  const auto size = A.Shape().Size();
  
  for (int64_t i = 0; i < size; i++, output++, input1++,input2++) {
    *output = 2*(*input1) + (*input2);
  }


  return Status::OK();
}

}  // namespace onnxruntime

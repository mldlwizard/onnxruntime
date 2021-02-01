#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/providers/cpu/element_wise_ranged_transform.h"

namespace onnxruntime {
namespace contrib {
template <typename T>
class TestOp : public OpKernel {
public:
  explicit TestOp(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* context) const override {
  auto X = context->Input<Tensor>(0);
  auto& dims = X->Shape();
  auto Y = context->Output(0, dims);
  auto X_Data = (X->template Data<T>());
  auto Y_Data = (Y->template MutableData<T>());

  for (int64_t i = 0, sz = dims.Size(); i < sz; ++i) {
    *Y_Data++ =2*( *X_Data++);
    }

    return Status::OK();
  }

};
}
}

#include "contrib_ops/cpu/TestOp.h"
namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    TestOp,
    kOnnxDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    TestOp<float>
);

}
}

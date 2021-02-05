#include "contrib_ops/cpu/HardSwish.h"
namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    HardSwish,
    kOnnxDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float,double>()),
    HardSwish
);

}
}

//
// Created by sam on 7/8/24.
//

#include "scalar_vector.h"
#include <roughpy/device_support/host_kernel.h>

#include <roughpy/devices/kernel.h>

using namespace rpy;
using namespace rpy::scalars;

namespace params = rpy::devices::params;

namespace {

struct UnaryVectorKernelFn {
};

class UnaryOperationKernel : public devices::HostKernel<
                                     UnaryVectorKernelFn,
                                     params::ResultBuffer<params::T1>,
                                     params::Buffer<params::T1>,
                                     params::Operator<params::T1>>
{

public:
    UnaryOperationKernel() : HostKernel("vector_unary_op") {}
};

using UnaryKernelSignature = devices::StandardKernelSignature<
        params::ResultBuffer<params::T1>,
        params::Buffer<params::T1>,
        params::Operator<params::T1>>;

using UnaryInplaceKernelSignature = devices::StandardKernelSignature<
        params::ResultBuffer<params::T1>,
        params::Operator<params::T1>>;

static UnaryOperationKernel uop_kernel{};

}// namespace

void UnaryVectorOperation::eval(
        ScalarVector& destination,
        const ScalarVector& source,
        const ops::Operator& op
) const
{}

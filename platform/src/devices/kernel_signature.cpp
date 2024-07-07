//
// Created by sam on 7/2/24.
//

#include "kernel.h"
#include "standard_kernel_arguments.h"


using namespace rpy;
using namespace rpy::devices;


KernelSignature::~KernelSignature() = default;

std::unique_ptr<KernelArguments> KernelSignature::new_binding() const
{
    return std::make_unique<StandardKernelArguments>(*this);
}

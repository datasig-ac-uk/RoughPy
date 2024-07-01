//
// Created by sam on 01/07/24.
//

#include "kernel.h"

using namespace rpy;
using namespace rpy::devices;

namespace ops = rpy::devices::operators;

KernelArguments::~KernelArguments() = default;

dimn_t KernelArguments::true_num_args() const noexcept
{
    return this->num_args();
}

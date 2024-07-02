//
// Created by sam on 01/07/24.
//

#include "kernel.h"


#include "host_device.h"

using namespace rpy;
using namespace rpy::devices;



Device KernelSpec::get_device() const noexcept
{
    (void) this;
    return get_host_device();
}


Slice<const TypePtr> KernelSpec::get_types() const noexcept
{
    return p_kernel_args->get_types();
}

//
// Created by sam on 4/17/24.
//

#include "kernels/kernel.h"

#include <roughpy/core/types.h>

using namespace rpy;
using namespace rpy::devices;


optional<Kernel> algebra::dtl::get_kernel(
        string_view kernel_name,
        string_view suffix,
        const Device& device
)
{
    auto name = string(kernel_name) + '_' + string(suffix);
    return device->get_kernel(name);
}

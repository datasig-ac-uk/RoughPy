//
// Created by sam on 7/8/24.
//

#include "scalar_vector.h"

using namespace rpy;
using namespace rpy::scalars;

VectorOperation::~VectorOperation() = default;

void VectorOperation::resize_destination(ScalarVector& arg, dimn_t new_size)
        const
{
    if (arg.dimension() < new_size) { arg.resize_base_dim(new_size); }
}

optional<devices::Kernel> VectorOperation::get_kernel(
        devices::Device device,
        string_view base_name,
        Slice<const Type* const> types
) const
{
    // implementation goes here
    string name(base_name);
    for (const auto& tp : types) {
        name += '_';
        name += tp->id();
    }
    return device->get_kernel(name);
}

devices::KernelLaunchParams
VectorOperation::get_launch_params(const devices::KernelArguments& arguments
) const noexcept
{
    /*
     * There are lots of naive things we can do here, but I think the most
     * sensible default is to simply use the max size of arguments in the block
     */

    auto sizes = arguments.get_sizes();
    auto min_size = sizes.empty() ? 0 : *ranges::min_element(sizes);

    devices::KernelLaunchParams params(devices::Size3{min_size, 1, 1});

    return params;
}

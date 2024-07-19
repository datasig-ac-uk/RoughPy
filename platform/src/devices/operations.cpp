//
// Created by sam on 7/11/24.
//

#include "operation.h"

#include "device_handle.h"
#include "kernel.h"

#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;

optional<Kernel> Operation::get_kernel(const KernelArguments& args) const
{
    auto lookup_name = string_join('_', m_kernel_name);
    for (const auto& type : args.get_types()) {
        lookup_name += '_';
        lookup_name += type->id();
    }

    return args.get_device()->get_kernel(lookup_name);
}

optional<Event> Operation::eval(
        Queue& queue,
        const KernelLaunchParams& params,
        const KernelArguments& args
) const
{
    if (const auto kernel = get_kernel(args)) {
        return kernel->launch_async_in_queue(queue, params, args);
    }
    return {};
}

using T1ResultBuffer = params::ResultBuffer<params::T1>;
using T1Buffer = params::Buffer<params::T1>;
using T1Operator = params::Operator<params::T1>;

RPY_MAKE_STANDARD_OPERATION_DECLARE(
        AdditionOperation,
        ROUGHPY_DEVICES_EXPORT,
        T1ResultBuffer,
        T1Buffer,
        T1Buffer,
        T1Operator
);

RPY_MAKE_STANDARD_OPERATION_IMPL(AdditionOperation, "addition")
{
    auto fn = [](Buffer& out,
                 const Buffer& left,
                 const Buffer& right,
                 const operators::Operator& op) {
        auto dim = rpy::ranges::min(left.size(), right.size());

        // for (dimn_t i = 0; i < dim; ++i) { out = op(left[i], right[i]); }
    };

    using Binder
            = ArgumentBinder<typename signature_t::ParamsList, decltype(fn)>;

    Binder::eval(fn, params, args);

    return EventStatus::CompletedSuccessfully;
}

// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 11/10/23.
//

#include "devices/event.h"
#include "devices/kernel.h"
#include "devices/queue.h"

#include "devices/device_handle.h"

#include <roughpy/core/helpers.h>

using namespace rpy;
using namespace rpy::devices;

namespace rpy {
namespace devices {
namespace dtl {

template class RPY_DLL_EXPORT ObjectBase<KernelInterface, Kernel>;
}
}// namespace devices
}// namespace rpy

string Kernel::name() const
{
    if (!impl()) { return ""; }
    return impl()->name();
}

dimn_t Kernel::num_args() const
{
    if (!impl()) { return 0; }
    return impl()->num_args();
}

Event Kernel::launch_async_in_queue(
        Queue& queue,
        const KernelLaunchParams& params,
        Slice<KernelArgument> args
)
{
    if (!impl() || !params.has_work()) { return Event(); }

    auto nargs = impl()->num_args();
    if (nargs != args.size()) {
        RPY_THROW(
                std::runtime_error,
                "kernel '" + impl()->name()
                        + "' called with incorrect number of arguments: "
                          "expected "
                        + std::to_string(nargs) + " arguments but got "
                        + std::to_string(args.size())
        );
    }

    if (!queue.is_default() && queue.device() != device()) {
        RPY_THROW(
                std::invalid_argument,
                "the queue provided is not a valid queue for this kernel"
        );
    }

    return impl()->launch_kernel_async(queue, params, args);
}
EventStatus Kernel::launch_sync_in_queue(
        Queue& queue,
        const KernelLaunchParams& params,
        Slice<KernelArgument> args
)
{
    auto event = launch_async_in_queue(queue, params, args);
    event.wait();
    return event.status();
}
Event Kernel::launch_async(
        const KernelLaunchParams& params,
        Slice<KernelArgument> args
)
{
    auto queue = device()->get_default_queue();
    return launch_async_in_queue(queue, params, args);
}
EventStatus Kernel::launch_sync(
        const KernelLaunchParams& params,
        Slice<KernelArgument> args
)
{
    auto queue = device()->get_default_queue();
    return launch_sync_in_queue(queue, params, args);
}

std::vector<bitmask_t>
Kernel::construct_work_mask(const KernelLaunchParams& params)
{
    RPY_DBG_ASSERT(params.has_work());
    std::vector<bitmask_t> result;

    auto total_work = params.total_work_size();
    auto num_masks = round_up_divide(total_work, CHAR_BIT * sizeof(bitmask_t));
    result.reserve(num_masks);
    result.resize(num_masks - 1, ~bitmask_t(0));

    auto underflow = num_masks * CHAR_BIT * sizeof(bitmask_t) - total_work;
    auto final_mask = ~((bitmask_t(1) << (underflow + 1)) - 1);
    result.push_back(final_mask);

    return result;
}

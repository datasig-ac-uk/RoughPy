// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 11/10/23.
//


#include <roughpy/device/kernel.h>
#include <roughpy/device/event.h>
#include <roughpy/device/queue.h>



using namespace rpy;
using namespace rpy::device;


string_view Kernel::name() const {
    if (interface() == nullptr || content() == nullptr) {
        return "";
    }
    return interface()->name(content());
}

dimn_t Kernel::num_args() const
{
    if (interface() == nullptr || content() == nullptr) {
        return 0;
    }
    return interface()->num_args(content());
}

Event Kernel::launch_async(
        Queue& queue,
        Slice<void*> args,
        Slice<dimn_t> arg_sizes,
        const KernelLaunchParams& params
)
{
    if (interface() == nullptr || content() == nullptr) {
        return Event(nullptr, nullptr);
    }

    auto nargs = interface()->num_args(content());
    if (nargs != args.size() || nargs != arg_sizes.size()) {
        RPY_THROW(std::runtime_error, "incorrect number of arguments provided");
    }

    return interface()->launch_kernel_async(content(), queue, args,
                                            arg_sizes, params);
}
EventStatus Kernel::launch_sync(
        Queue& queue,
        Slice<void*> args,
        Slice<dimn_t> arg_sizes,
        const KernelLaunchParams& params
)
{
    auto event = launch_async(queue, args, arg_sizes, params);
    event.wait();
    return event.status();
}

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
// Created by sam on 25/08/23.
//

#include <roughpy/device/kernel.h>

#include <roughpy/device/queue.h>

using namespace rpy;
using namespace rpy::device;

string_view KernelInterface::name(void* content) const
{
    return std::string_view();
}
dimn_t KernelInterface::num_args(void* content) const { return 0; }

string_view Kernel::name() const {
    if (interface() == nullptr || content() == nullptr) {
        return {};
    }
    return interface()->name(content());
}
dimn_t Kernel::num_args() const {
    if (interface() == nullptr || content() == nullptr) {
        return 0;
    }
    return interface()->num_args(content());
}

void KernelInterface::launch_kernel_sync(
        void* content, Queue queue, Slice<void*> args, Slice<dimn_t> arg_sizes,
        const KernelLaunchParams& params
) const
{
    auto event = launch_kernel_async(content, queue, args, arg_sizes, params);
    event.wait();
}

Event KernelInterface::launch_kernel_async(
        void* content, Queue queue, Slice<void*> args, Slice<dimn_t> arg_sizes,
        const KernelLaunchParams& params
) const
{
    return Event();
}

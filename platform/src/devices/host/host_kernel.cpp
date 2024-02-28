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
// Created by user on 16/10/23.
//

#include "host_kernel.h"
#include "host_device_impl.h"

#include <roughpy/core/helpers.h>

using namespace rpy;
using namespace rpy::devices;



CPUKernel::CPUKernel(fallback_kernel_t fallback, uint32_t nargs, string name)
    : m_fallback(fallback),
      m_name(std::move(name)),
      m_nargs(nargs)
{}



string CPUKernel::name() const { return m_name; }
dimn_t CPUKernel::num_args() const { return m_nargs; }
Event CPUKernel::launch_kernel_async(
        Queue& queue,
        const KernelLaunchParams& params,
        Slice<KernelArgument> args
)
{
    return Event();
}
std::unique_ptr<rpy::devices::dtl::InterfaceBase> CPUKernel::clone() const
{
    return std::make_unique<CPUKernel>(m_fallback, m_nargs, m_name);
}
Device CPUKernel::device() const noexcept { return CPUDeviceHandle::get(); }

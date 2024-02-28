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

#ifndef ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_KERNEL_H_
#define ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_KERNEL_H_

#include "devices/kernel.h"


namespace rpy {
namespace devices {

class CPUKernel : public dtl::RefCountBase<KernelInterface>
{
    using fallback_kernel_t = void (*)(void**, Size3 work_size) noexcept;

    fallback_kernel_t m_fallback;
    string m_name;
    uint32_t m_nargs;




public:
    CPUKernel(fallback_kernel_t fallback, uint32_t nargs, string name);

    std::unique_ptr<InterfaceBase> clone() const override;
    Device device() const noexcept override;

    string name() const override;
    dimn_t num_args() const override;
    Event launch_kernel_async(
            Queue& queue,
            const KernelLaunchParams& params,
            Slice<KernelArgument> args
    ) override;
};

}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_KERNEL_H_

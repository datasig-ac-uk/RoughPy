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

#ifndef ROUGHPY_DEVICE_KERNEL_H_
#define ROUGHPY_DEVICE_KERNEL_H_

#include "core.h"
#include "device_handle.h"
#include "device_object_base.h"
#include "event.h"
#include "kernel_arg.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/types.h>

namespace rpy {
namespace devices {

class KernelLaunchParams
{
    Size3 m_work_dims;
    Dim3 m_group_size;
    optional<Dim3> m_offsets;

public:
    RPY_NO_DISCARD bool has_work() const noexcept;

    RPY_NO_DISCARD Size3 total_work_dims() const noexcept;

    RPY_NO_DISCARD dimn_t total_work_size() const noexcept;

    RPY_NO_DISCARD dsize_t num_dims() const noexcept;

    RPY_NO_DISCARD Dim3 num_work_groups() const noexcept;

    RPY_NO_DISCARD Size3 underflow_of_groups() const noexcept;

    KernelLaunchParams();
};

class RPY_EXPORT KernelInterface : public dtl::InterfaceBase
{

public:
    RPY_NO_DISCARD virtual string name() const;

    RPY_NO_DISCARD virtual dimn_t num_args() const;

    RPY_NO_DISCARD virtual Event launch_kernel_async(
            Queue& queue,
            Slice<void*> args,
            Slice<dimn_t> arg_sizes,
            const KernelLaunchParams& params
    );

    virtual void init_args(std::vector<KernalArgument*>& args) const;
};

class RPY_EXPORT Kernel : public dtl::ObjectBase<KernelInterface, Kernel>
{
    using base_t = dtl::ObjectBase<KernelInterface, Kernel>;

    std::vector<KernalArgument*> m_args;

public:
    using base_t::base_t;

    RPY_NO_DISCARD bool is_nop() const noexcept { return !p_impl; }

    RPY_NO_DISCARD string name() const;

    RPY_NO_DISCARD dimn_t num_args() const;

    RPY_NO_DISCARD Event launch_async(
            Queue& queue,
            Slice<void*> args,
            Slice<dimn_t> arg_sizes,
            const KernelLaunchParams& params
    );

    RPY_NO_DISCARD EventStatus launch_sync(
            Queue& queue,
            Slice<void*> args,
            Slice<dimn_t> arg_sizes,
            const KernelLaunchParams& params
    );

    RPY_NO_DISCARD static std::vector<bitmask_t>
    construct_work_mask(const KernelLaunchParams& params);

    template <typename... Args>
    void operator()(const KernelLaunchParams& params, Args&&... args);
};

template <typename... Args>
void Kernel::operator()(const KernelLaunchParams& params, Args&&... args)
{
    std::vector<void*> arg_p{arg_to_pointer(args)...};
    std::vector<dimn_t> arg_s{sizeof(Args)...};

    Queue default_queue;
    auto status = launch_sync(default_queue, arg_p, arg_s, params);
    RPY_DBG_ASSERT(status == EventStatus::CompletedSuccessfully);
}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_KERNEL_H_

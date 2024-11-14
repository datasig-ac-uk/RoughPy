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
#include "device_object_base.h"
#include "event.h"
#include "kernel_arg.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/types.h>

namespace rpy {
namespace devices {

class ROUGHPY_PLATFORM_EXPORT KernelLaunchParams
{
    Size3 m_work_dims;
    Dim3 m_group_size;
    optional<Dim3> m_offsets;

public:

    explicit KernelLaunchParams(Size3 work_dims)
        : m_work_dims(work_dims),
          m_group_size(),
          m_offsets()
    {}

    explicit KernelLaunchParams(Size3 work_dims, Dim3 group_size)
            : m_work_dims(work_dims),
          m_group_size(group_size),
          m_offsets()
    {}

    RPY_NO_DISCARD bool has_work() const noexcept;

    RPY_NO_DISCARD Size3 total_work_dims() const noexcept;

    RPY_NO_DISCARD dimn_t total_work_size() const noexcept;

    RPY_NO_DISCARD dsize_t num_dims() const noexcept;

    RPY_NO_DISCARD Dim3 num_work_groups() const noexcept;

    RPY_NO_DISCARD Size3 underflow_of_groups() const noexcept;

    RPY_NO_DISCARD Dim3 work_groups() const noexcept;

    KernelLaunchParams();
};

class ROUGHPY_PLATFORM_EXPORT KernelInterface : public dtl::InterfaceBase
{
public:

    using object_t = Kernel;

    RPY_NO_DISCARD virtual string name() const;

    RPY_NO_DISCARD virtual dimn_t num_args() const;

    RPY_NO_DISCARD virtual Event launch_kernel_async(
            Queue& queue,
            const KernelLaunchParams& params,
            Slice<KernelArgument> args
    );

};

#ifdef RPY_PLATFORM_WINDOWS
#  ifdef RoughPy_Platform_EXPORTS
namespace dtl {
extern template class ObjectBase<KernelInterface, Kernel>;
}
#  else
namespace dtl {
template class RPY_DLL_IMPORT ObjectBase<KernelInterface, Kernel>;
}
#  endif
#else
namespace dtl {
extern template class ROUGHPY_PLATFORM_EXPORT
        ObjectBase<KernelInterface, Kernel>;
}
#endif

class ROUGHPY_PLATFORM_EXPORT Kernel
    : public dtl::ObjectBase<KernelInterface, Kernel>
{
    using base_t = dtl::ObjectBase<KernelInterface, Kernel>;

    std::vector<KernelArgument*> m_args;

public:
    using base_t::base_t;

    RPY_NO_DISCARD bool is_nop() const noexcept { return is_null(); }

    RPY_NO_DISCARD string name() const;

    RPY_NO_DISCARD dimn_t num_args() const;

    RPY_NO_DISCARD Event launch_async_in_queue(
            Queue& queue,
            const KernelLaunchParams& params,
            Slice<KernelArgument> args
    );

    RPY_NO_DISCARD EventStatus launch_sync_in_queue(
            Queue& queue,
            const KernelLaunchParams& params,
            Slice<KernelArgument> args
    );

    RPY_NO_DISCARD Event
    launch_async(const KernelLaunchParams& params, Slice<KernelArgument> args);

    RPY_NO_DISCARD EventStatus
    launch_sync(const KernelLaunchParams& params, Slice<KernelArgument> args);

    RPY_NO_DISCARD static std::vector<bitmask_t>
    construct_work_mask(const KernelLaunchParams& params);

    template <typename... Args>
    void operator()(const KernelLaunchParams& params, Args&&... args);
};

template <typename... Args>
void Kernel::operator()(const KernelLaunchParams& params, Args&&... args)
{
    KernelArgument kargs[] = {KernelArgument(args)...};
    auto status = launch_sync(params, kargs);
    RPY_CHECK(status == EventStatus::CompletedSuccessfully);
}

template <typename... Args>
RPY_NO_DISCARD Event
launch_async(Kernel kernel, const KernelLaunchParams& params, Args... args)
{
    KernelArgument kargs[] = {KernelArgument(args)...};
    return kernel.launch_async(params, kargs);
}

template <typename... Args>
EventStatus
launch_sync(Kernel kernel, const KernelLaunchParams& params, Args... args)
{
    KernelArgument kargs[] = {KernelArgument(args)...};
    return kernel.launch_sync(params, kargs);
}

template <typename... Args>
RPY_NO_DISCARD Event launch_async(
        Kernel kernel,
        Queue& queue,
        const KernelLaunchParams& params,
        Args... args
)
{
    KernelArgument kargs[] = {KernelArgument(args)...};
    return kernel.launch_async_in_queue(queue, params, kargs);
}

template <typename... Args>
EventStatus launch_sync(
        Kernel kernel,
        Queue& queue,
        const KernelLaunchParams& params,
        Args... args
)
{
    KernelArgument kargs[] = {KernelArgument(args)...};
    return kernel.launch_sync_in_queue(queue, params, kargs);
}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_KERNEL_H_

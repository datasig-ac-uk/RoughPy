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
// Created by user on 16/10/23.
//

#ifndef ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_KERNEL_H_
#define ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_KERNEL_H_

#include "devices/kernel.h"
#include "devices/buffer.h"
#include "devices/memory_view.h"

namespace rpy {
namespace devices {


template <typename... Ts>
using raw_kfn_ptr = void (*)(Ts...);


class CPUKernel : public dtl::RefCountBase<KernelInterface>
{
    using wrapped_kernel_t = std::function<void(const KernelLaunchParams&, Slice<KernelArgument>)>;

    wrapped_kernel_t m_kernel;
    string m_name;
    uint32_t m_nargs;

public:
    template <typename... Ts>
    CPUKernel(raw_kfn_ptr<Ts...> fn, string name);

    template <typename... Ts>
    CPUKernel(std::function<void(Ts...)> fn, string name);


    Device device() const noexcept override;

    string name() const override;
    dimn_t num_args() const override;
    Event launch_kernel_async(
            Queue& queue,
            const KernelLaunchParams& params,
            Slice<KernelArgument> args
    ) override;
};

namespace dtl {

template <typename T>
class ConvertedKernelArgument
{
    T* p_data;
public:

    explicit ConvertedKernelArgument(KernelArgument& arg);

    operator T() { return *p_data; }
};

template <typename T>
ConvertedKernelArgument<T>::ConvertedKernelArgument(KernelArgument& arg)
{
    RPY_CHECK(arg.info() == type_info<T>());
    p_data = arg.const_pointer();
}


template <typename T>
class ConvertedKernelArgument<T*>
{
    T* p_data;
    MutableMemoryView m_view;

public:

    explicit ConvertedKernelArgument(KernelArgument& arg);

    operator T*() { return p_data; }

};

template <typename T>
ConvertedKernelArgument<T*>::ConvertedKernelArgument(KernelArgument& arg)
{
    RPY_CHECK(arg.info() == type_info<T>());
    RPY_CHECK(arg.is_buffer() && !arg.is_const());
    auto& buffer = arg.buffer();

    if (buffer.device() != get_host_device()) {
        m_view = buffer.map_raw(buffer.size(), 0);
        p_data = reinterpret_cast<T*>(m_view.raw_ptr(0));
    } else {
        p_data = buffer.ptr();
    }
}


template <typename T>
class ConvertedKernelArgument<const T*>
{
    const T* p_data;
    MemoryView m_view;
public:

    explicit ConvertedKernelArgument(KernelArgument& arg);

    operator const T*() { return p_data; }

};

template <typename T>
ConvertedKernelArgument<const T*>::ConvertedKernelArgument(KernelArgument& arg)
{
    RPY_CHECK(arg.info() == type_info<T>());
    RPY_CHECK(arg.is_buffer() && !arg.is_const());
    auto& buffer = arg.buffer();

    if (buffer.device() != get_host_device()) {
        m_view = buffer.map();
        p_data = reinterpret_cast<T*>(m_view.raw_ptr(0));
    } else {
        p_data = buffer.ptr();
    }
}

/*
 * The problem we need to solve now is how to invoke a function that takes a
 * normal list of arguments, given an array of KernelArguments. To do this,
 * we're going to use the power of variadic templates to do the unpacking.
 *
 * We'd like to do something very simple like the following
 *
 * template <typename... Args>
 * void invoke(void (*)(Args... args) fn, Slice<KernelArgument> args, ...) {
 *      auto* aptr = args.data();
 *      fn(cast<Args>(aptr++)...);
 * }
 *
 * However, this invokes undefined behaviour: the increment operators are not
 * guaranteed to happen in order. To get around this, we have to explicitly
 * index into the argument array by first making an index template pack that
 * can be unpacked at the same time to disambiguate the increment operations.
 * This is based on answers on SO: https://stackoverflow.com/a/11044592/9225581,
 * https://stackoverflow.com/a/10930078/9225581, and the referenced article
 * http://loungecpp.wikidot.com/tips-and-tricks%3aindices
 *
 */

template <size_t... Is>
struct Indices {
};

template <size_t N, size_t... Is>
struct BuildIndices {
    using type = typename BuildIndices<N - 1, N - 1, Is...>::type;
};

template <size_t... Is>
struct BuildIndices<0, Is...> {
    using type = Indices<Is...>;
};

template <typename... Ts>
using indices_for = BuildIndices<sizeof...(Ts)>;

template <typename... Ts, size_t... Is>
RPY_INLINE_ALWAYS void invoke_kernel_inner(
        std::function<void(Ts...)> fn,
        Indices<Is...>,
        Slice<KernelArgument> args
)
{
    fn(ConvertedKernelArgument<Ts>(args[Is])...);
}

template <typename... Ts>
RPY_INLINE_ALWAYS void invoke_kernel(std::function<void(Ts...)> fn, Slice<KernelArgument> args)
{
    invoke_kernel_inner(std::move(fn), indices_for<Ts...>(), args);
}



template <typename... Ts, size_t... Is>
RPY_INLINE_ALWAYS void invoke_kernel_inner(
        raw_kfn_ptr<Ts...> fn,
        Indices<Is...>,
        Slice<KernelArgument> args
)
{
    fn(ConvertedKernelArgument<Ts>(args[Is])...);
}

template <typename... Ts>
RPY_INLINE_ALWAYS void invoke_kernel(raw_kfn_ptr<void(Ts...)> fn, Slice<KernelArgument> args)
{
    invoke_kernel_inner(fn, indices_for<Ts...>(), args);
}





}// namespace dtl

template <typename... Ts>
CPUKernel::CPUKernel(raw_kfn_ptr<Ts...> fn, string name)
    : m_kernel([fn](const KernelLaunchParams& params, Slice<KernelArgument> args) {
          dtl::invoke_kernel(fn, args);
      }),
      m_nargs(sizeof...(Ts)),
      m_name(std::move(name))
{}

template <typename... Ts>
CPUKernel::CPUKernel(std::function<void(Ts...)> fn, string name)
    : m_kernel([fn=std::move(fn)](const KernelLaunchParams& params, Slice<KernelArgument> args) {
          dtl::invoke_kernel(fn, args);
      }),
      m_nargs(sizeof...(Ts)),
      m_name(std::move(name))
{}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_KERNEL_H_

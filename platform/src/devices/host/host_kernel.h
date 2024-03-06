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
class ConvertedKernelArgument<T*>
{
    T* p_data;

public:

    explicit ConvertedKernelArgument(KernelArgument& arg);

    operator T*() { return p_data; }

};

template <typename T>
class ConvertedKernelArgument<const T*>
{
    const T* p_data;

public:

    explicit ConvertedKernelArgument(KernelArgument& arg);

    operator const T*() { return p_data; }

};

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

}// namespace dtl

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SRC_CPUDEVICE_CPU_KERNEL_H_

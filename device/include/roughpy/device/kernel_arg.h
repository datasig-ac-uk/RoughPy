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

#ifndef ROUGHPY_DEVICE_KERNEL_ARG_H_
#define ROUGHPY_DEVICE_KERNEL_ARG_H_

#include "core.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include "types.h"

namespace rpy {
namespace devices {

class RPY_EXPORT KernelArgument
{

public:
    virtual ~KernelArgument();

    virtual string name() const noexcept;
    virtual string type_string() const noexcept;

    virtual void set(Buffer& data);
    virtual void set(const Buffer& data);
    virtual void set(void* data, const TypeInfo& info);
    virtual void set(const void* data, const TypeInfo& info);

    void set(half data);
    void set(bfloat16 data);
    void set(float data);
    void set(double data);
    void set(const rational_scalar_type& data);
    void set(const rational_poly_scalar& data);

    template <typename I>
    enable_if_t<is_integral<I>::value> set(I data);
};

template <typename I>
enable_if_t<is_integral<I>::value> KernelArgument::set(I data)
{
    this->set(
            &data,
            {is_signed<I>::value ? TypeCode::Int : TypeCode::UInt,
             sizeof(I),
             1}
    );
}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_KERNEL_ARG_H_

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

#include "device_object_base.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>
#include <roughpy/core/slice.h>

#include "core.h"
#include "event.h"


namespace rpy { namespace device {


class RPY_EXPORT KernelInterface : public dtl::InterfaceBase {


public:

    RPY_NO_DISCARD
    virtual string_view name(void* content) const;

    RPY_NO_DISCARD
    virtual dimn_t num_args(void* content) const;

    RPY_NO_DISCARD
    virtual Event launch_kernel_async(void* content,
                                      Queue queue,
                                      Slice<void*> args,
                                      Slice<dimn_t> arg_sizes,
                                      const KernelLaunchParams& params) const;



    virtual void launch_kernel_sync(void* content,
                                    Queue queue,
                                    Slice<void*> args,
                                    Slice<dimn_t> arg_sizes,
                                    const KernelLaunchParams& params) const;



};


class RPY_EXPORT Kernel : public dtl::ObjectBase<KernelInterface, Kernel> {

public:

    RPY_NO_DISCARD
    string_view name() const;

    RPY_NO_DISCARD
    dimn_t num_args() const;

};

}}

#endif // ROUGHPY_DEVICE_KERNEL_H_

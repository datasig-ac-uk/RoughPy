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

#ifndef ROUGHPY_ALGEBRA_FREE_TENSOR_BUNDLE_INTERFACE_H_
#define ROUGHPY_ALGEBRA_FREE_TENSOR_BUNDLE_INTERFACE_H_

#include <roughpy/core/macros.h>

#include <roughpy/algebra/tensor_basis.h>
#include <roughpy/algebra/free_tensor.h>
#include <roughpy/algebra/free_tensor_bundle.h>


#include "algebra_bundle_interface.h"

RPY_WARNING_PUSH
RPY_GCC_DISABLE_WARNING(-Wattributes)

namespace rpy { namespace algebra {


RPY_TEMPLATE_EXTERN template class RPY_EXPORT_TEMPLATE
        BundleInterface<FreeTensorBundle, FreeTensor, FreeTensor>;



class RPY_EXPORT FreeTensorBundleInterface
    : public BundleInterface<FreeTensorBundle, FreeTensor, FreeTensor>
{
public:
    using algebra_interface_t
            = BundleInterface<FreeTensorBundle, FreeTensor, FreeTensor>;

    using algebra_interface_t::algebra_interface_t;

    RPY_NO_DISCARD virtual FreeTensorBundle exp() const = 0;
    RPY_NO_DISCARD virtual FreeTensorBundle log() const = 0;
    //    RPY_NO_DISCARD
    //    virtual FreeTensorBundle inverse() const = 0;
    RPY_NO_DISCARD virtual FreeTensorBundle antipode() const = 0;
    virtual void fmexp(const FreeTensorBundle& other) = 0;
};

}}

RPY_WARNING_POP
#endif // ROUGHPY_ALGEBRA_FREE_TENSOR_BUNDLE_INTERFACE_H_

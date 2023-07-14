// Copyright (c) 2023 RoughPy Developers. All rights reserved.
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

#ifndef ROUGHPY_ALGEBRA_FREE_TENSOR_H_
#define ROUGHPY_ALGEBRA_FREE_TENSOR_H_

#include "free_tensor_fwd.h"

#include "algebra_base_impl.h"
#include "algebra_bundle_base_impl.h"
#include "context.h"

namespace rpy {
namespace algebra {

RPY_WARNING_PUSH
RPY_GCC_DISABLE_WARNING(-Wattributes)

RPY_TEMPLATE_EXTERN template class RPY_EXPORT_TEMPLATE
        AlgebraInterface<FreeTensor, TensorBasis>;

RPY_TEMPLATE_EXTERN template class RPY_EXPORT_TEMPLATE
        AlgebraBase<FreeTensorInterface, FreeTensorImplementation>;

RPY_TEMPLATE_EXTERN template class RPY_EXPORT_TEMPLATE
        BundleInterface<FreeTensorBundle, FreeTensor, FreeTensor>;

RPY_TEMPLATE_EXTERN template class RPY_EXPORT_TEMPLATE AlgebraBundleBase<
        FreeTensorBundleInterface, FreeTensorBundleImplementation>;

RPY_WARNING_POP

template <>
RPY_EXPORT typename FreeTensor::basis_type
basis_setup_helper<FreeTensor>::get(const context_pointer& ctx);

template <>
RPY_EXPORT typename FreeTensorBundle::basis_type
basis_setup_helper<FreeTensorBundle>::get(const context_pointer& ctx);

}// namespace algebra
}// namespace rpy
#endif// ROUGHPY_ALGEBRA_FREE_TENSOR_H_

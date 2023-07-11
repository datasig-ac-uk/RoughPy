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

//
// Created by user on 06/03/23.
//

#include <roughpy/algebra/algebra_bundle_impl.h>
#include <roughpy/algebra/algebra_impl.h>
#include <roughpy/algebra/context.h>
#include <roughpy/algebra/shuffle_tensor.h>
#include <roughpy/platform/serialization.h>

namespace rpy {
namespace algebra {

template class RPY_EXPORT_INSTANTIATION
        AlgebraInterface<ShuffleTensor, TensorBasis>;

template class RPY_EXPORT_INSTANTIATION AlgebraBase<ShuffleTensorInterface>;

template class RPY_EXPORT_INSTANTIATION
        BundleInterface<ShuffleTensorBundle, ShuffleTensor, ShuffleTensor>;

template class RPY_EXPORT_INSTANTIATION
        AlgebraBundleBase<ShuffleTensorBundleInterface>;

template <>
typename ShuffleTensor::basis_type
basis_setup_helper<ShuffleTensor>::get(const context_pointer& ctx)
{
    return ctx->get_tensor_basis();
}

template <>
typename ShuffleTensorBundle::basis_type
basis_setup_helper<ShuffleTensorBundle>::get(const context_pointer& ctx)
{
    return ctx->get_tensor_basis();
}

}// namespace algebra
}// namespace rpy

#define RPY_SERIAL_IMPL_CLASSNAME rpy::algebra::ShuffleTensor

#include <roughpy/platform/serialization_instantiations.inl>

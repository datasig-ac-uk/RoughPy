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

#ifndef ROUGHPY_ALGEBRA_LIE_H_
#define ROUGHPY_ALGEBRA_LIE_H_

#include "lie_fwd.h"

#include <roughpy/core/macros.h>

#include "algebra_base_impl.h"
#include "algebra_bundle_base_impl.h"
#include "context.h"

namespace rpy {
namespace algebra {

RPY_WARNING_PUSH
RPY_GCC_DISABLE_WARNING(-Wattributes)
RPY_TEMPLATE_EXTERN template class RPY_EXPORT_TEMPLATE
        AlgebraInterface<Lie, LieBasis>;

RPY_TEMPLATE_EXTERN template class RPY_EXPORT_TEMPLATE
        AlgebraBase<LieInterface>;

RPY_TEMPLATE_EXTERN template class RPY_EXPORT_TEMPLATE
        BundleInterface<LieBundle, Lie, Lie>;

RPY_TEMPLATE_EXTERN template class RPY_EXPORT_TEMPLATE
        AlgebraBundleBase<LieBundleInterface>;

RPY_WARNING_POP

RPY_SERIAL_SERIALIZE_FN_IMPL(Lie) { RPY_SERIAL_SERIALIZE_BASE(base_t); }

RPY_SERIAL_SERIALIZE_FN_IMPL(LieBundle) { RPY_SERIAL_SERIALIZE_BASE(base_t); }

template <>
RPY_EXPORT typename Lie::basis_type
basis_setup_helper<Lie>::get(const context_pointer& ctx);

template <>
RPY_EXPORT typename LieBundle::basis_type
basis_setup_helper<LieBundle>::get(const context_pointer& ctx);

}// namespace algebra
}// namespace rpy
#endif// ROUGHPY_ALGEBRA_LIE_H_

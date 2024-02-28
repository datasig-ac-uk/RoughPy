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

#ifndef ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_H_
#define ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_H_

#include "algebra_fwd.h"
#include "tensor_basis.h"
#include "algebra_base.h"
#include "interfaces/shuffle_tensor_interface.h"

#include <roughpy/platform/serialization.h>

RPY_WARNING_PUSH
RPY_GCC_DISABLE_WARNING(-Wattributes)
RPY_MSVC_DISABLE_WARNING(4661)

namespace rpy {
namespace algebra {

#ifdef RPY_PLATFORM_WINDOWS
#  ifdef RPY_COMPILING_DLL
extern template class AlgebraBase<ShuffleTensorInterface>;
#  else
template class RPY_DLL_IMPORT AlgebraBase<ShuffleTensorInterface>;
#  endif
#else
extern template class ROUGHPY_ALGEBRA_EXPORT
        AlgebraBase<ShuffleTensorInterface>;
#endif

class ROUGHPY_ALGEBRA_EXPORT ShuffleTensor : public AlgebraBase<ShuffleTensorInterface>
{
    using base_t = AlgebraBase<ShuffleTensorInterface>;

public:
    static constexpr AlgebraType s_alg_type = AlgebraType::ShuffleTensor;

    using base_t::base_t;

    RPY_SERIAL_SERIALIZE_FN();
};

#ifdef RPY_COMPILING_ALGEBRA
RPY_SERIAL_EXTERN_SERIALIZE_CLS_BUILD(ShuffleTensor)
#else
RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMP(ShuffleTensor)
#endif

RPY_SERIAL_SERIALIZE_FN_IMPL(ShuffleTensor)
{
    RPY_SERIAL_SERIALIZE_BASE(base_t);
}



template <>
ROUGHPY_ALGEBRA_EXPORT typename ShuffleTensor::basis_type
basis_setup_helper<ShuffleTensor>::get(const context_pointer& ctx);



}// namespace algebra
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::algebra::ShuffleTensor,
        rpy::serial::specialization::member_serialize
)

RPY_WARNING_POP
#endif// ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_H_

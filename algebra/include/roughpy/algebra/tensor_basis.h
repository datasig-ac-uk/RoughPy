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

#ifndef ROUGHPY_ALGEBRA_TENSOR_BASIS_H_
#define ROUGHPY_ALGEBRA_TENSOR_BASIS_H_

#include "algebra_fwd.h"

#include "basis.h"

namespace rpy {
namespace algebra {

class ROUGHPY_ALGEBRA_EXPORT TensorBasisInterface
    : public make_basis_interface<
              TensorBasisInterface, rpy::key_type, OrderedBasisInterface,
              WordLikeBasisInterface>
{
public:
    ~TensorBasisInterface() override;
};

#ifdef RPY_PLATFORM_WINDOWS
#  ifdef RPY_COMPILING_DLL
extern template class Basis<TensorBasisInterface>;
#  else
template class RPY_DLL_IMPORT Basis<TensorBasisInterface>;
#  endif
#else
extern template class ROUGHPY_ALGEBRA_EXPORT Basis<TensorBasisInterface>;
#endif

        using TensorBasis = Basis<TensorBasisInterface>;

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_TENSOR_BASIS_H_

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

#ifndef ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_INTERFACE_H_
#define ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_INTERFACE_H_

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/algebra/tensor_basis.h>
#include <roughpy/algebra/shuffle_tensor.h>
#include "algebra_interface.h"

namespace rpy {
namespace algebra {

#ifdef RPY_PLATFORM_WINDOWS
#  ifdef RPY_COMPILING_DLL
extern template class AlgebraInterface<ShuffleTensor, TensorBasis>;
#  else
template class RPY_DLL_IMPORT AlgebraInterface<ShuffleTensor, TensorBasis>;
#  endif
#else
extern template class ROUGHPY_ALGEBRA_EXPORT
        AlgebraInterface<ShuffleTensor, TensorBasis>;
#endif

class ROUGHPY_ALGEBRA_EXPORT ShuffleTensorInterface
    : public AlgebraInterface<ShuffleTensor, TensorBasis>
{
    using base_t = AlgebraInterface<ShuffleTensor, TensorBasis>;

public:
    using base_t::base_t;
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_INTERFACE_H_

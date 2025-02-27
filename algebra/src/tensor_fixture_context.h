// Copyright (c) 2024 RoughPy Developers. All rights reserved.
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

#ifndef ROUGHPY_ALGEBRA_SRC_TENSOR_FIXTURE_CONTEXT_H
#define ROUGHPY_ALGEBRA_SRC_TENSOR_FIXTURE_CONTEXT_H

#include "roughpy/core/ranges.h"
#include "roughpy/core/types.h"
#include "roughpy/algebra/context.h"
#include "roughpy/scalars/scalar_types.h"

namespace rpy {
namespace algebra {
namespace testing {


//! Helper object wrapping building of free tensors in unit tests
class TensorFixtureContext
{
public:
    const scalars::ScalarType* rational_poly_tp;
    const rpy::algebra::context_pointer context;

public:
    TensorFixtureContext(deg_t width, deg_t depth);

    //! Create free tensor with all coeffs 1 of width and depth and given char
    RPY_NO_DISCARD FreeTensor make_ones_tensor(
        char indeterminate_char
    ) const;

    //! Create free tensor with all coeffs N of width and depth and given char
    RPY_NO_DISCARD FreeTensor make_ns_tensor(
        char indeterminate_char,
        scalars::rational_scalar_type n
    ) const;

    //! Create free tensor with each coeff constructed from make_coeff_fn
    //! defaulting to default tensor data size. Lambda signature is:
    //!     make_coeff_fn(size_t index) -> scalars::rational_poly_scalar
    template <typename MakeCoeffFn>
    RPY_NO_DISCARD FreeTensor make_tensor(
        MakeCoeffFn&& make_coeff_fn
    ) const
    {
        using namespace rpy::scalars;

        // Construct and allocate a rational polynomial
        VectorConstructionData cons_data{
            KeyScalarArray(rational_poly_tp),
            VectorType::Dense
        };
        const dimn_t size = context->tensor_size(context->depth());
        cons_data.data.allocate_scalars(size);

        // Delegate the construction of each basis with coefficient
        auto slice = cons_data.data.as_mut_slice<rational_poly_scalar>();
        for (auto&& [i, coeff] : views::enumerate(slice)) {
            coeff = make_coeff_fn(i);
        }

        FreeTensor result = context->construct_free_tensor(cons_data);
        return result;
    }
};


} // namespace testing
} // namespace algebra
} // namespace rpy

#endif // ROUGHPY_ALGEBRA_SRC_TENSOR_FIXTURE_CONTEXT_H

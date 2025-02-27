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

#include "tensor_fixture_context.h"

namespace rpy {
namespace algebra {
namespace testing {


TensorFixtureContext::TensorFixtureContext(deg_t width, deg_t depth) :
    rational_poly_tp{*scalars::ScalarType::of<devices::rational_poly_scalar>()},
    context{rpy::algebra::get_context(width, depth, rational_poly_tp)}
{
}


std::vector<dimn_t> TensorFixtureContext::basis_starts() const
{
    std::vector<dimn_t> result;

    auto basis = context->get_tensor_basis();
    for (size_t i = 0, depth = basis.depth(); i <= depth + 2; ++i) {
        int start = basis.start_of_degree(i);
        result.push_back(start);
    }

    return result;
}


std::vector<dimn_t> TensorFixtureContext::basis_sizes() const
{
    std::vector<dimn_t> result;

    auto basis = context->get_tensor_basis();
    for (size_t i = 0, depth = basis.depth(); i <= depth; ++i) {
        int size = (!i) ? 1 : basis.start_of_degree(i - 1) * basis.width() + 2;
        result.push_back(size);
    }

    return result;
}


FreeTensor TensorFixtureContext::make_ones_tensor(
    char indeterminate_char
) const
{
    FreeTensor result = make_tensor([indeterminate_char](size_t i) {
        auto key = scalars::indeterminate_type(indeterminate_char, i);
        auto coeff = scalars::rational_poly_scalar(key, scalars::rational_scalar_type(1));
        return coeff;
    });
    return result;
}


FreeTensor TensorFixtureContext::make_ns_tensor(
    char indeterminate_char,
    scalars::rational_scalar_type n
) const
{
    FreeTensor result = make_tensor([indeterminate_char, n](size_t i) {
        auto key = scalars::indeterminate_type(indeterminate_char, i);
        auto coeff = scalars::rational_poly_scalar(key, n);
        return coeff;
    });
    return result;
}


} // namespace testing
} // namespace algebra
} // namespace rpy

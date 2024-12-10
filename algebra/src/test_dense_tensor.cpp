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

#include "tensor_fixture.h"

namespace {

using namespace rpy;
using namespace rpy::algebra;
using namespace rpy::algebra::testing;
using namespace scalars;

using DenseTensorFixture = TensorFixture;

} // namespace

TEST_F(DenseTensorFixture, test_construct_from_context)
{
    FreeTensor t(context);
    ASSERT_EQ(t.context(), context);
    ASSERT_EQ(t.width(), 2);
    ASSERT_EQ(t.depth(), 5);
}

TEST_F(DenseTensorFixture, test_copy_construct)
{
    FreeTensor original = make_ns_tensor('x', 3);
    FreeTensor t(original);
    ASSERT_EQ(t, original);
}

TEST_F(DenseTensorFixture, test_move_construct)
{
    FreeTensor original = make_ns_tensor('x', 5);
    FreeTensor expected = original; // Copy assert compare value before move
    FreeTensor t(std::move(original));
    ASSERT_EQ(original.context(), nullptr); // Original's impl invalidated
    ASSERT_EQ(t, expected);
}

TEST_F(DenseTensorFixture, test_copy_assign)
{
}

TEST_F(DenseTensorFixture, test_move_assign)
{
}

TEST_F(DenseTensorFixture, test_context)
{
}

TEST_F(DenseTensorFixture, test_basis_type)
{
}

TEST_F(DenseTensorFixture, test_basis)
{
}

TEST_F(DenseTensorFixture, test_borrow)
{
}

TEST_F(DenseTensorFixture, test_borrow_mut)
{
}

TEST_F(DenseTensorFixture, test_add)
{
    FreeTensor lhs = make_ns_tensor('x', 2);
    FreeTensor rhs = make_ns_tensor('y', 3);
    FreeTensor result = lhs.add(rhs);

    FreeTensor expected = make_tensor([](size_t i) {
        auto lhs_coeff = rational_poly_scalar(indeterminate_type('x', i), 2);
        auto rhs_coeff = rational_poly_scalar(indeterminate_type('y', i), 3);
        return lhs_coeff + rhs_coeff;
    });

    ASSERT_EQ(result, expected);
}

TEST_F(DenseTensorFixture, test_sub)
{
    FreeTensor lhs = make_ns_tensor('x', 3);
    FreeTensor rhs = make_ns_tensor('y', 5);
    FreeTensor result = lhs.sub(rhs);

    FreeTensor expected = make_tensor([](size_t i) {
        auto lhs_coeff = rational_poly_scalar(indeterminate_type('x', i), 3);
        auto rhs_coeff = rational_poly_scalar(indeterminate_type('y', i), 5);
        return lhs_coeff - rhs_coeff;
    });

    ASSERT_EQ(result, expected);
}

TEST_F(DenseTensorFixture, test_mul)
{
}

TEST_F(DenseTensorFixture, test_dimension)
{
}

TEST_F(DenseTensorFixture, test_size)
{
}

TEST_F(DenseTensorFixture, test_is_zero)
{
}

TEST_F(DenseTensorFixture, test_width)
{
}

TEST_F(DenseTensorFixture, test_depth)
{
}

TEST_F(DenseTensorFixture, test_degree)
{
}

TEST_F(DenseTensorFixture, test_storage_type)
{
}

TEST_F(DenseTensorFixture, test_coeff_type)
{
}

TEST_F(DenseTensorFixture, test_operator_index)
{
}

TEST_F(DenseTensorFixture, test_operator_const_index)
{
}

TEST_F(DenseTensorFixture, test_iterators)
{
    // Tensor has key values 0..N for N basis
    FreeTensor t = make_tensor([](size_t i) {
        auto key = indeterminate_type('x', i);
        auto coeff = rational_poly_scalar(key, 1) * i;
        return coeff;
    });

    // Check that each value is 0..N as expected
    size_t expected_index = 0;
    for (auto it = t.begin(); it != t.end(); ++it) {
        ASSERT_EQ(it->key(), expected_index);
        // TODO check it->value() is expected_index
        ++expected_index;
    }
    ASSERT_EQ(expected_index, t.size());
}

TEST_F(DenseTensorFixture, test_dense_data)
{
}

TEST_F(DenseTensorFixture, test_uminus)
{
    FreeTensor lhs = make_ones_tensor('x');
    FreeTensor result = lhs.uminus();

    FreeTensor expected = make_tensor([](size_t i) {
        auto key = indeterminate_type('x', i);
        auto coeff = rational_poly_scalar(key, 1);
        return -coeff;
    });

    ASSERT_EQ(result, expected);
}

TEST_F(DenseTensorFixture, test_smul)
{
    FreeTensor lhs = make_ns_tensor('x', 5);
    Scalar rhs(7);
    FreeTensor result = lhs.smul(rhs);

    FreeTensor expected = make_tensor([](size_t i) {
        auto key = indeterminate_type('x', i);
        auto coeff = rational_poly_scalar(key, 5);
        return coeff * 7;
    });

    ASSERT_EQ(result, expected);
}

TEST_F(DenseTensorFixture, test_sdiv)
{
}

TEST_F(DenseTensorFixture, test_add_inplace)
{
}

TEST_F(DenseTensorFixture, test_sub_inplace)
{
}

TEST_F(DenseTensorFixture, test_mul_inplace)
{
}

TEST_F(DenseTensorFixture, test_smul_inplace)
{
}

TEST_F(DenseTensorFixture, test_sdiv_inplace)
{
}

TEST_F(DenseTensorFixture, test_add_scal_mul)
{
}

TEST_F(DenseTensorFixture, test_sub_scal_mul)
{
}

TEST_F(DenseTensorFixture, test_add_scal_div)
{
}

TEST_F(DenseTensorFixture, test_sub_scal_div)
{
}

TEST_F(DenseTensorFixture, test_add_mul)
{
}

TEST_F(DenseTensorFixture, test_sub_mul)
{
}

TEST_F(DenseTensorFixture, test_mul_smul)
{
}

TEST_F(DenseTensorFixture, test_mul_sdiv)
{
}

TEST_F(DenseTensorFixture, test_operator_equals)
{
}

TEST_F(DenseTensorFixture, test_print)
{
}

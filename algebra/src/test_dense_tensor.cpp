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

TEST_F(DenseTensorFixture, test_accessors)
{
    // Construct a tensor with zeros after a certain point; size() will not
    // take stock of zero elements, but dimension() does.
    const size_t index_zero_after = 32;
    FreeTensor lhs = make_tensor([index_zero_after](size_t i) {
        auto adjusted_index = (i <= index_zero_after) ? i : 0;
        auto key = indeterminate_type('x', adjusted_index);
        auto coeff = rational_poly_scalar(key, adjusted_index);
        return coeff;
    });

    ASSERT_EQ(lhs.context(), context);
    ASSERT_EQ(lhs.basis(), context->get_tensor_basis());
    ASSERT_EQ(lhs.dimension(), 63);
    ASSERT_EQ(lhs.size(), index_zero_after);
    ASSERT_EQ(lhs.width(), 2);
    ASSERT_EQ(lhs.depth(), 5);
    ASSERT_EQ(lhs.degree(), 5);
}

TEST_F(DenseTensorFixture, test_borrow)
{
}

TEST_F(DenseTensorFixture, test_borrow_mut)
{
}

TEST_F(DenseTensorFixture, test_add)
{
    FreeTensor lhs = make_ones_tensor('x');
    FreeTensor rhs = make_ones_tensor('y');
    FreeTensor result = lhs.add(rhs);

    FreeTensor expected = make_tensor([](size_t i) {
        auto lhs_coeff = rational_poly_scalar(indeterminate_type('x', i), 1);
        auto rhs_coeff = rational_poly_scalar(indeterminate_type('y', i), 1);
        auto coeff = lhs_coeff + rhs_coeff;
        return coeff;
    });

    ASSERT_EQ(result, expected);

    // Test again with inplace version
    result = lhs;
    result.add_inplace(rhs);
    ASSERT_EQ(result, expected);
}

TEST_F(DenseTensorFixture, test_sub)
{
    FreeTensor lhs = make_ones_tensor('x');
    FreeTensor rhs = make_ones_tensor('y');
    FreeTensor result = lhs.sub(rhs);

    FreeTensor expected = make_tensor([](size_t i) {
        auto lhs_coeff = rational_poly_scalar(indeterminate_type('x', i), 1);
        auto rhs_coeff = rational_poly_scalar(indeterminate_type('y', i), 1);
        auto coeff = lhs_coeff - rhs_coeff;
        return coeff;
    });

    ASSERT_EQ(result, expected);

    // Test again with inplace version
    result = lhs;
    result.sub_inplace(rhs);
    ASSERT_EQ(result, expected);
}

TEST_F(DenseTensorFixture, test_mul)
{
}


TEST_F(DenseTensorFixture, test_is_zero)
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
        auto coeff = rational_poly_scalar(key, i);
        return coeff;
    });

    // Check that each value is 0..N as expected
    size_t expected_index = 0;
    for (auto it = t.begin(); it != t.end(); ++it) {
        ASSERT_EQ(it->key(), expected_index);

        // Type must be constructed from coeff_type for valid comparison
        auto expected_key = indeterminate_type('x', expected_index);
        auto expected_coeff = rational_poly_scalar(expected_key, expected_index);
        auto coeff_as_scalar = Scalar(t.coeff_type(), expected_coeff);
        ASSERT_EQ(it->value(), coeff_as_scalar);
        ++expected_index;
    }

    ASSERT_EQ(expected_index, t.dimension());
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
        auto coeff = -rational_poly_scalar(key, 1);
        return coeff;
    });

    ASSERT_EQ(result, expected);
}

TEST_F(DenseTensorFixture, test_smul)
{
    FreeTensor lhs = make_ones_tensor('x');
    Scalar rhs(7);
    FreeTensor result = lhs.smul(rhs);

    FreeTensor expected = make_tensor([](size_t i) {
        auto key = indeterminate_type('x', i);
        auto coeff = rational_poly_scalar(key, 7);
        return coeff;
    });

    ASSERT_EQ(result, expected);

    // Test again with inplace version
    result = lhs;
    result.smul_inplace(rhs);
    ASSERT_EQ(result, expected);
}

TEST_F(DenseTensorFixture, test_sdiv)
{
    FreeTensor lhs = make_ones_tensor('x');
    Scalar rhs(11);
    FreeTensor result = lhs.sdiv(rhs);

    FreeTensor expected = make_tensor([](size_t i) {
        auto key = indeterminate_type('x', i);
        auto coeff = rational_poly_scalar(key, 1) / 11;
        return coeff;
    });

    ASSERT_EQ(result, expected);

    // Test again with inplace version
    result = lhs;
    result.sdiv_inplace(rhs);
    ASSERT_EQ(result, expected);
}

TEST_F(DenseTensorFixture, test_mul_inplace)
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

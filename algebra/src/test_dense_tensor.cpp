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

using TestDenseTensor = TensorFixture;

} // namespace


TEST_F(TestDenseTensor, TestContextStarts)
{
    // Sanity check that the context has right degree start values for a W=2,
    // D=2 test, so multiplications tests are correct.
    auto starts = builder->basis_starts();
    std::vector<dimn_t> expected_starts{0, 1, 3, 7, 15, 31, 63, 127};
    ASSERT_EQ(starts, expected_starts);
}


TEST_F(TestDenseTensor, TestContextSizes)
{
    // Similar to TestContextStarts, check size values so multiplications tests are correct.
    auto sizes = builder->basis_sizes();
    std::vector<dimn_t> expected_sizes{1, 2, 4, 8, 16, 32};
    ASSERT_EQ(sizes, expected_sizes);
}


TEST_F(TestDenseTensor, TestConstructFromContext)
{
    FreeTensor tensor(builder->context);
    ASSERT_EQ(tensor.context(), builder->context);
    ASSERT_EQ(tensor.width(), 2);
    ASSERT_EQ(tensor.depth(), 5);
}

TEST_F(TestDenseTensor, TestCopyConstruct)
{
    FreeTensor original = builder->make_ns_tensor('x', 3);
    FreeTensor copied(original);
    ASSERT_EQ(copied, original);
}

TEST_F(TestDenseTensor, TestMoveConstruct)
{
    FreeTensor original = builder->make_ns_tensor('x', 5);
    FreeTensor expected = original; // Copy assert compare value before move
    FreeTensor moved(std::move(original));
    ASSERT_EQ(original.context(), nullptr); // Original's impl invalidated
    ASSERT_TENSOR_EQ(moved, expected);
}

TEST_F(TestDenseTensor, TestCopyAssign)
{
}

TEST_F(TestDenseTensor, TestMoveAssign)
{
}

TEST_F(TestDenseTensor, TestAccessors)
{
    // Construct a tensor with zeros after a certain point; size() will not
    // take stock of zero elements, but dimension() does.
    const size_t index_zero_after = 32;
    FreeTensor lhs = builder->make_tensor([index_zero_after](size_t i) {
        auto adjusted_index = (i <= index_zero_after) ? i : 0;
        auto key = indeterminate_type('x', adjusted_index);
        auto coeff = rational_poly_scalar(key, adjusted_index);
        return coeff;
    });

    ASSERT_EQ(lhs.context(), builder->context);
    ASSERT_EQ(lhs.basis(), builder->context->get_tensor_basis());
    ASSERT_EQ(lhs.dimension(), 63);
    ASSERT_EQ(lhs.size(), index_zero_after);
    ASSERT_EQ(lhs.width(), 2);
    ASSERT_EQ(lhs.depth(), 5);
    ASSERT_EQ(lhs.degree(), 5);
}

TEST_F(TestDenseTensor, TestBorrow)
{
}

TEST_F(TestDenseTensor, TestBorrowMut)
{
}

TEST_F(TestDenseTensor, TestAdd)
{
    FreeTensor lhs = builder->make_ones_tensor('x');
    FreeTensor rhs = builder->make_ones_tensor('y');
    FreeTensor result = lhs.add(rhs);

    FreeTensor expected = builder->make_tensor([](size_t i) {
        auto lhs_coeff = rational_poly_scalar(indeterminate_type('x', i), 1);
        auto rhs_coeff = rational_poly_scalar(indeterminate_type('y', i), 1);
        auto coeff = lhs_coeff + rhs_coeff;
        return coeff;
    });

    ASSERT_TENSOR_EQ(result, expected);

    // Test again with in-place version
    result = lhs;
    auto inplace_copy = result.add_inplace(rhs);
    ASSERT_TENSOR_EQ(result, expected);
    ASSERT_TENSOR_EQ(result, inplace_copy);
}

TEST_F(TestDenseTensor, TestSub)
{
    FreeTensor lhs = builder->make_ones_tensor('x');
    FreeTensor rhs = builder->make_ones_tensor('y');
    FreeTensor result = lhs.sub(rhs);

    FreeTensor expected = builder->make_tensor([](size_t i) {
        auto lhs_coeff = rational_poly_scalar(indeterminate_type('x', i), 1);
        auto rhs_coeff = rational_poly_scalar(indeterminate_type('y', i), 1);
        auto coeff = lhs_coeff - rhs_coeff;
        return coeff;
    });

    ASSERT_TENSOR_EQ(result, expected);

    // Test again with in-place version
    result = lhs;
    auto inplace_copy = result.sub_inplace(rhs);
    ASSERT_TENSOR_EQ(result, expected);
    ASSERT_TENSOR_EQ(result, inplace_copy);
}


TEST_F(TestDenseTensor, TestMulSameContext)
{
    FreeTensor lhs = builder->make_ones_tensor('x');
    FreeTensor rhs = builder->make_ones_tensor('y');
    FreeTensor result = lhs.mul(rhs);

    FreeTensor expected = builder->make_mul_tensor([](dimn_t left_k, dimn_t right_k){
        auto coeff_x = rational_poly_scalar(indeterminate_type('x', left_k), 1);
        auto coeff_y = rational_poly_scalar(indeterminate_type('y', right_k), 1);
        return coeff_x * coeff_y;
    });

    ASSERT_TENSOR_EQ(result, expected);
}

TEST_F(TestDenseTensor, TestMulDiffWidthsError)
{
}

TEST_F(TestDenseTensor, TestMulDiffLhsDepthSmallerOk)
{
}

TEST_F(TestDenseTensor, TestMulDiffRhsDepthSmallerOk)
{
}

TEST_F(TestDenseTensor, TestIsZero)
{
}

TEST_F(TestDenseTensor, TestStorageType)
{
}

TEST_F(TestDenseTensor, TestCoeffType)
{
}

TEST_F(TestDenseTensor, TestOperatorIndex)
{
}

TEST_F(TestDenseTensor, TestOperatorConstIndex)
{
}

TEST_F(TestDenseTensor, TestIterators)
{
    // Tensor has key values 0...N for N basis
    FreeTensor t = builder->make_tensor([](size_t i) {
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

TEST_F(TestDenseTensor, TestDenseData)
{
}

TEST_F(TestDenseTensor, TestUminus)
{
    FreeTensor lhs = builder->make_ones_tensor('x');
    FreeTensor result = lhs.uminus();

    FreeTensor expected = builder->make_tensor([](size_t i) {
        auto key = indeterminate_type('x', i);
        auto coeff = -rational_poly_scalar(key, 1);
        return coeff;
    });

    ASSERT_TENSOR_EQ(result, expected);
}

TEST_F(TestDenseTensor, TestSmul)
{
    FreeTensor lhs = builder->make_ones_tensor('x');
    Scalar rhs(7);
    FreeTensor result = lhs.smul(rhs);

    FreeTensor expected = builder->make_tensor([](size_t i) {
        auto key = indeterminate_type('x', i);
        auto coeff = rational_poly_scalar(key, 7);
        return coeff;
    });

    ASSERT_TENSOR_EQ(result, expected);

    // Test again with in-place version
    result = lhs;
    auto inplace_copy = result.smul_inplace(rhs);
    ASSERT_TENSOR_EQ(result, expected);
    ASSERT_TENSOR_EQ(result, inplace_copy);
}

TEST_F(TestDenseTensor, TestSdiv)
{
    FreeTensor lhs = builder->make_ones_tensor('x');
    Scalar rhs(11);
    FreeTensor result = lhs.sdiv(rhs);

    FreeTensor expected = builder->make_tensor([](size_t i) {
        auto key = indeterminate_type('x', i);
        auto coeff = rational_poly_scalar(key, 1) / 11;
        return coeff;
    });

    ASSERT_TENSOR_EQ(result, expected);

    // Test again with in-place version
    result = lhs;
    auto inplace_copy = result.sdiv_inplace(rhs);
    ASSERT_TENSOR_EQ(result, expected);
    ASSERT_TENSOR_EQ(result, inplace_copy);
}

TEST_F(TestDenseTensor, TestMulInplace)
{
}

TEST_F(TestDenseTensor, TestAddScalMulSameIndeterminate)
{
    FreeTensor lhs = builder->make_ns_tensor('x', 2);
    FreeTensor rhs = builder->make_ns_tensor('x', 3);
    Scalar scale{5};

    // SAXPY eqivalent 2x + (3x * 5) = 17x
    FreeTensor expected = builder->make_ns_tensor('x', 17);

    FreeTensor result = lhs.add_scal_mul(rhs, scale);
    ASSERT_TENSOR_EQ(result, expected);

    // add_scal_mul also modifies in-place
    ASSERT_TENSOR_EQ(lhs, expected);
}

TEST_F(TestDenseTensor, TestAddScalMulDiffIndeterminate)
{
    FreeTensor lhs = builder->make_ns_tensor('x', 2);
    FreeTensor rhs = builder->make_ns_tensor('y', 3);
    Scalar scale{5};

    // SAXPY eqivalent 2x + (3y * 5) = 2x + 15y
    FreeTensor expected = builder->make_tensor([](size_t i) {
        auto x_coeff = rational_poly_scalar(indeterminate_type('x', i), 2);
        auto y_coeff = rational_poly_scalar(indeterminate_type('y', i), 15);
        return x_coeff + y_coeff;
    });

    FreeTensor result = lhs.add_scal_mul(rhs, scale);
    ASSERT_TENSOR_EQ(result, expected);

    // add_scal_mul also modifies in-place
    ASSERT_TENSOR_EQ(lhs, expected);
}

TEST_F(TestDenseTensor, TestAddScalMulDiffSize)
{
    FreeTensor lhs = builder->make_ones_tensor('x');

    TensorFixtureContext diff_builder{1, 5}; // Exception: different context width
    FreeTensor rhs = diff_builder.make_ones_tensor('x');
    Scalar scale{1};

    FreeTensor expected = lhs; // Value should not change before exception

    ASSERT_THROW(
        (void)lhs.add_scal_mul(rhs, scale),
        std::runtime_error
    );

    ASSERT_TENSOR_EQ(lhs, expected);
}

TEST_F(TestDenseTensor, TestAddScalMulZeroLhs)
{
    FreeTensor lhs = builder->make_ns_tensor('x', 0);
    FreeTensor rhs = builder->make_ns_tensor('y', 2);
    Scalar scale{3};

    // SAXPY eqivalent 0x + (2y * 3) = 6y
    FreeTensor expected = builder->make_tensor([](size_t i) {
        auto coeff = rational_poly_scalar(indeterminate_type('y', i), 6);
        return coeff;
    });

    FreeTensor result = lhs.add_scal_mul(rhs, scale);
    ASSERT_TENSOR_EQ(result, expected);

    // add_scal_mul also modifies in-place
    ASSERT_TENSOR_EQ(lhs, expected);
}


TEST_F(TestDenseTensor, TestSubScalMul)
{
    FreeTensor lhs = builder->make_ns_tensor('x', 5);
    FreeTensor rhs = builder->make_ns_tensor('y', 3);
    Scalar scale{2};

    // SAXPY eqivalent 5x - (3y * 2) = 5x - 6y
    FreeTensor expected = builder->make_tensor([](size_t i) {
        auto x_coeff = rational_poly_scalar(indeterminate_type('x', i), 5);
        auto y_coeff = rational_poly_scalar(indeterminate_type('y', i), -6);
        return x_coeff + y_coeff;
    });

    FreeTensor result = lhs.sub_scal_mul(rhs, scale);
    ASSERT_TENSOR_EQ(result, expected);

    // sub_scal_mul also modifies in-place
    ASSERT_TENSOR_EQ(lhs, expected);
}

TEST_F(TestDenseTensor, TestAddScalDivRational)
{
    FreeTensor lhs = builder->make_ns_tensor('x', 5);
    FreeTensor rhs = builder->make_ns_tensor('y', 3);
    Scalar scale{2};

    // SAXPY eqivalent 5x + (3y / 2)
    FreeTensor expected = builder->make_tensor([](size_t i) {
        auto x_coeff = rational_poly_scalar(indeterminate_type('x', i), 5);
        auto y_coeff = rational_poly_scalar(indeterminate_type('y', i), rational_scalar_type(3, 2));
        return x_coeff + y_coeff;
    });

    FreeTensor result = lhs.add_scal_div(rhs, scale);

    ASSERT_TENSOR_EQ(result, expected);

    // add_scal_div also modifies in-place
    ASSERT_TENSOR_EQ(lhs, expected);
}

TEST_F(TestDenseTensor, TestAddScalDivZero)
{
    FreeTensor lhs = builder->make_ns_tensor('x', 1);
    FreeTensor rhs = builder->make_ns_tensor('y', 1);
    Scalar scale{0}; // Exception: divide by zero

    FreeTensor expected = lhs; // Value should not change before exception

    ASSERT_THROW(
        (void)lhs.add_scal_div(rhs, scale),
        std::invalid_argument
    );

    ASSERT_TENSOR_EQ(lhs, expected);
}

TEST_F(TestDenseTensor, TestAddScalDivInvalid)
{
    // An invalid scalar for division is an indeterminate
    FreeTensor lhs = builder->make_ns_tensor('x', 1);
    FreeTensor rhs = builder->make_ns_tensor('y', 1);
    Scalar scale{
        builder->context->ctype(),
        rational_poly_scalar(indeterminate_type('a', 1), 1)
    };

    FreeTensor expected = lhs; // Value should not change before exception

    ASSERT_THROW(
        (void)lhs.add_scal_div(rhs, scale),
        std::runtime_error
    );

    ASSERT_TENSOR_EQ(lhs, expected);
}

TEST_F(TestDenseTensor, TestAddMul)
{
}

TEST_F(TestDenseTensor, TestSubMul)
{
}

TEST_F(TestDenseTensor, TestMulSmul)
{
}

TEST_F(TestDenseTensor, TestMulSdiv)
{
}

TEST_F(TestDenseTensor, TestOperatorEquals)
{
}

TEST_F(TestDenseTensor, TestPrint)
{
}

#include <vector>

#include <gtest/gtest.h>

#include "roughpy_compute/dense/views.hpp"

#include "shuffle_tensor_adjoint_mul.hpp"
#include "shuffle_tensor_product.hpp"

#include "roughpy_compute/testing/tensor_helpers.hpp"

using namespace rpy::compute;

class ShuffleTensorAdjointMulTests
    : public ::testing::Test,
      public rpy::compute::testing::PolynomialTensorHelper
{
    using Helper = rpy::compute::testing::PolynomialTensorHelper;
    using Basis = typename Helper::Basis;
    using Index = typename Basis::Index;

    std::vector<Index> tensor_begins;

protected:
    static constexpr int32_t width = 3;
    static constexpr int32_t depth = 4;

    void SetUp() override
    {
        tensor_begins.resize(depth + 2);
        for (int32_t i = 1; i <= depth + 1; ++i) {
            tensor_begins[i] = 1 + width * tensor_begins[i - 1];
        }
    }

    [[nodiscard]]
    Basis get_basis() const noexcept
    {
        return {tensor_begins.data(), width, depth};
    }

    [[nodiscard]]
    static Scalar pairing(std::vector<Scalar> const& lhs, std::vector<Scalar> const& rhs)
    {
        Scalar result{0};
        auto const size = lhs.size();
        for (size_t i = 0; i < size; ++i) {
            result += lhs[i] * rhs[i];
        }
        return result;
    }

    [[nodiscard]]
    static std::vector<Scalar> linear_combo(
        std::vector<Scalar> const& lhs,
        Scalar const& lhs_scale,
        std::vector<Scalar> const& rhs,
        Scalar const& rhs_scale
    )
    {
        std::vector<Scalar> result(lhs.size());
        for (size_t i = 0; i < lhs.size(); ++i) {
            result[i] = lhs_scale * lhs[i] + rhs_scale * rhs[i];
        }
        return result;
    }

    [[nodiscard]]
    static std::vector<Scalar> apply_shuffle(
        Basis const& basis,
        std::vector<Scalar> const& lhs,
        std::vector<Scalar> const& rhs
    )
    {
        std::vector<Scalar> out(basis.size());
        DenseTensorView<Scalar*> out_view(out.data(), basis);
        DenseTensorView<Scalar const*> lhs_view(lhs.data(), basis);
        DenseTensorView<Scalar const*> rhs_view(rhs.data(), basis);
        basic::v1::st_fma(out_view, lhs_view, rhs_view);
        return out;
    }

    [[nodiscard]]
    static std::vector<Scalar> apply_adj_mul(
        Basis const& basis,
        std::vector<Scalar> const& op,
        std::vector<Scalar> const& arg
    )
    {
        std::vector<Scalar> out(basis.size());
        DenseTensorView<Scalar*> out_view(out.data(), basis);
        DenseTensorView<Scalar const*> op_view(op.data(), basis);
        DenseTensorView<Scalar const*> arg_view(arg.data(), basis);
        basic::v1::st_adj_mul(out_view, op_view, arg_view);
        return out;
    }

    using Helper::make_scalar;
    using Helper::make_tensor;
    using Helper::word_to_idx_fn;
};

TEST_F(ShuffleTensorAdjointMulTests, SatisfiesAdjointPairingCriterion)
{
    auto const basis = get_basis();
    auto const to_idx = word_to_idx_fn(basis);

    auto const a = make_tensor('a', basis, to_idx);
    auto const b = make_tensor('b', basis, to_idx);
    auto const x = make_tensor('x', basis, to_idx);

    auto const m_a_b = apply_shuffle(basis, a, b);
    auto const m_a_adj_x = apply_adj_mul(basis, a, x);

    auto const lhs = pairing(m_a_b, x);
    auto const rhs = pairing(b, m_a_adj_x);

    EXPECT_EQ(lhs, rhs);
}

TEST_F(ShuffleTensorAdjointMulTests, IsBilinearInOperatorAndArgument)
{
    auto const basis = get_basis();
    auto const to_idx = word_to_idx_fn(basis);

    auto const a1 = make_tensor('a', basis, to_idx);
    auto const a2 = make_tensor('b', basis, to_idx);
    auto const x1 = make_tensor('x', basis, to_idx);
    auto const x2 = make_tensor('y', basis, to_idx);

    auto const alpha = make_scalar({{{{'p', 1}}, 2, 1}});
    auto const beta = make_scalar({{{{'q', 2}}, 3, 1}});
    auto const gamma = make_scalar({{{{'r', 3}}, 5, 1}});
    auto const delta = make_scalar({{{{'s', 4}}, 7, 1}});

    auto const a = linear_combo(a1, alpha, a2, beta);
    auto const x = linear_combo(x1, gamma, x2, delta);

    auto const lhs = apply_adj_mul(basis, a, x);

    auto const ax11 = apply_adj_mul(basis, a1, x1);
    auto const ax12 = apply_adj_mul(basis, a1, x2);
    auto const ax21 = apply_adj_mul(basis, a2, x1);
    auto const ax22 = apply_adj_mul(basis, a2, x2);

    std::vector<Scalar> rhs(basis.size());
    for (size_t i = 0; i < rhs.size(); ++i) {
        rhs[i] = (alpha * gamma) * ax11[i]
               + (alpha * delta) * ax12[i]
               + (beta * gamma) * ax21[i]
               + (beta * delta) * ax22[i];
    }

    EXPECT_EQ(lhs, rhs);
}

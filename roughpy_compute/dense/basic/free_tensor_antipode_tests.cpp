#include <vector>

#include <gtest/gtest.h>

#include "roughpy_compute/dense/views.hpp"

#include "free_tensor_antipode.hpp"

#include "roughpy_compute/testing/tensor_helpers.hpp"

using namespace rpy::compute;

class FreeTensorAntipodeTests
    : public ::testing::Test,
      public rpy::compute::testing::PolynomialTensorHelper
{
    using Helper = rpy::compute::testing::PolynomialTensorHelper;
    using Basis = typename Helper::Basis;
    using Index = typename Basis::Index;
    using Degree = typename Basis::Degree;

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

    using Helper::make_tensor;
    using Helper::word_to_idx_fn;
};

TEST_F(FreeTensorAntipodeTests, ExplicitSignedReverseAction)
{
    auto const basis = get_basis();
    auto const to_idx = word_to_idx_fn(basis);

    auto const arg = make_tensor('a', basis, to_idx);
    std::vector<Scalar> out(basis.size());
    std::vector<Scalar> expected(basis.size());

    DenseTensorView<Scalar*> out_view(out.data(), basis);
    DenseTensorView<Scalar const*> arg_view(arg.data(), basis);

    basic::v1::ft_antipode(
        out_view,
        arg_view,
        basic::v1::BasicAntipodeConfig{},
        basic::v1::DefaultSigner{}
    );

    for (auto degree = decltype(basis.depth){0}; degree <= basis.depth; ++degree) {
        auto const level_begin = basis.degree_begin[degree];
        auto const level_end = basis.degree_begin[degree + 1];
        auto const level_size = level_end - level_begin;
        auto const is_odd = (degree % 2) != 0;

        for (std::ptrdiff_t i = 0; i < level_size; ++i) {
            auto const rev = basic::v1::reverse_index<std::ptrdiff_t, int32_t>(
                i,
                static_cast<std::ptrdiff_t>(basis.width),
                static_cast<int32_t>(degree)
            );
            auto& slot = expected[level_begin + rev];
            slot = is_odd ? -arg[level_begin + i] : arg[level_begin + i];
        }
    }

    EXPECT_EQ(out, expected);
}

TEST_F(FreeTensorAntipodeTests, AntipodeIsInvolution)
{
    auto const basis = get_basis();
    auto const to_idx = word_to_idx_fn(basis);

    auto const arg = make_tensor('a', basis, to_idx);
    std::vector<Scalar> tmp(basis.size());
    std::vector<Scalar> out(basis.size());

    DenseTensorView<Scalar*> tmp_view(tmp.data(), basis);
    DenseTensorView<Scalar*> out_view(out.data(), basis);
    DenseTensorView<Scalar const*> arg_view(arg.data(), basis);
    DenseTensorView<Scalar const*> tmp_const_view(tmp.data(), basis);

    basic::v1::ft_antipode(
        tmp_view,
        arg_view,
        basic::v1::BasicAntipodeConfig{},
        basic::v1::DefaultSigner{}
    );

    basic::v1::ft_antipode(
        out_view,
        tmp_const_view,
        basic::v1::BasicAntipodeConfig{},
        basic::v1::DefaultSigner{}
    );

    EXPECT_EQ(out, arg);
}

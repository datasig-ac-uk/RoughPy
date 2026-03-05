#include <vector>

#include <gtest/gtest.h>

#include "roughpy_compute/common/operations.hpp"
#include "roughpy_compute/dense/views.hpp"

#include "shuffle_tensor_product.hpp"

#include "roughpy_compute/testing/tensor_helpers.hpp"

using namespace rpy::compute;

class ShuffleTensorProductTests
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

    template <typename ShuffleFn>
    void run_shuffle_and_check(ShuffleFn&& shuffle_fn)
    {
        auto const basis = get_basis();
        auto const to_idx = word_to_idx_fn(basis);

        auto out = make_tensor('c', basis, to_idx);
        auto const lhs = make_tensor('a', basis, to_idx);
        auto const rhs = make_tensor('b', basis, to_idx);

        DenseTensorView<Scalar*> out_view(out.data(), basis);
        DenseTensorView<Scalar const*> lhs_view(lhs.data(), basis);
        DenseTensorView<Scalar const*> rhs_view(rhs.data(), basis);

        shuffle_fn(out_view, lhs_view, rhs_view);

        std::vector<Scalar> expected;
        expected.reserve(basis.size());

        for_each_word(
            basis,
            [&expected, &to_idx](auto const& word) {
                auto const degree = word.size();

                auto entry = make_scalar({{{{'c', to_idx(word)}}, 1, 1}});

                auto const n_masks = size_t{1} << degree;
                for (size_t mask = 0; mask < n_masks; ++mask) {
                    auto const lhs_idx = to_idx(word, mask, 1);
                    auto const rhs_idx = to_idx(word, mask, 0);

                    entry += make_scalar({
                        {{{'a', lhs_idx}, {'b', rhs_idx}}, 1, 1}
                    });
                }

                expected.emplace_back(std::move(entry));
            }
        );

        EXPECT_EQ(out, expected);
    }

    using Helper::for_each_word;
    using Helper::make_scalar;
    using Helper::make_tensor;
    using Helper::word_to_idx_fn;
};

TEST_F(ShuffleTensorProductTests, ShuffleTensorFmaV1ExplicitAccumulation)
{
    run_shuffle_and_check(
        [](auto out, auto lhs, auto rhs) { basic::v1::st_fma(out, lhs, rhs); }
    );
}

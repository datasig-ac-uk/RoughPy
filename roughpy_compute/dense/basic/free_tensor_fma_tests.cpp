
#include <vector>

#include <gtest/gtest.h>

#include "roughpy_compute/dense/views.hpp"
#include "roughpy_compute/common/operations.hpp"

#include "free_tensor_fma.hpp"

#include "roughpy_compute/testing/tensor_helpers.hpp"


using namespace rpy::compute;


class FreeTensorFmaTests : public ::testing::Test, public rpy::compute::testing::PolynomialTensorHelper
{
    using Helper = rpy::compute::testing::PolynomialTensorHelper;

    std::vector<size_t> tensor_begins;

protected:
    static constexpr int32_t width = 3;
    static constexpr int32_t depth = 3;

    void SetUp() override
    {
        tensor_begins.resize(depth+2);
        for (int32_t i=1; i <= depth+1; ++i) {
            tensor_begins[i] = 1 + width * tensor_begins[i-1];
        }
    }

    [[nodiscard]]
    Basis get_basis() const noexcept
    {
        return { tensor_begins.data(), width, depth };
    }

    using Helper::make_scalar;
    using Helper::make_tensor;
    using Helper::word_to_idx_fn;
    using Helper::for_each_word;
};




TEST_F(FreeTensorFmaTests, FreeTensorFmaIdentity)
{
    auto const basis = get_basis();
    auto to_idx = word_to_idx_fn(basis);

    auto out = make_tensor('c', basis, to_idx);
    const auto lhs = make_tensor('a', basis, to_idx);
    const auto rhs = make_tensor('b', basis, to_idx);

    DenseTensorView<Scalar*> out_view(out.data(), basis);
    DenseTensorView<Scalar const*> lhs_view(lhs.data(), basis);
    DenseTensorView<Scalar const*> rhs_view(rhs.data(), basis);


    basic::ft_fma(out_view, lhs_view, rhs_view);

    std::vector<Scalar> expected;
    expected.reserve(basis.size());
    for_each_word(basis,
                  [&expected, &to_idx](auto const& word) {
                      auto degree = word.size();

                      auto& entry = expected.emplace_back(make_scalar({{{{'c', to_idx(word)}}, 1, 1}}));

                      for (size_t mid=0; mid <= degree; ++mid) {
                          auto lhs_idx = to_idx(word.data(), mid);
                          auto rhs_idx = to_idx(word.data() + mid, degree - mid);

                          entry += make_scalar({
                                  {{{'a', lhs_idx}, {'b', rhs_idx}}, 1, 1}
                          });
                      }
                  }
    );

    EXPECT_EQ(out, expected);
}
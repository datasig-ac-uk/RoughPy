
#include <vector>

#include <gtest/gtest.h>


#include "roughpy_compute/dense/views.hpp"

#include "vector_inplace_addition.hpp"

#include "roughpy_compute/testing/tensor_helpers.hpp"


using namespace rpy::compute;


class VectorInplaceAdditionTests : public ::testing::Test, public rpy::compute::testing::PolynomialTensorHelper
{
    using Helper = rpy::compute::testing::PolynomialTensorHelper;
    using Basis = typename Helper::Basis;
    using Index = typename Basis::Index;

    std::vector<Index> tensor_begins;
protected:


    using Scalar = typename Helper::Scalar;
    using Monomial = typename Scalar::key_type;
    using Indeterminant = typename Monomial::letter_type;
    using Rational = typename Scalar::scalar_type;

    using typename Helper::Degree;

    template <typename Ptr_>
    using VectorView = DenseVectorView<Ptr_, Basis>;

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



TEST_F(VectorInplaceAdditionTests, VectorAdditionIdentityOps)
{
    auto basis = get_basis();

    auto to_idx = word_to_idx_fn(basis);

    auto lhs = make_tensor('a', basis, to_idx);
    const auto rhs = make_tensor('b', basis, to_idx);

    VectorView<Scalar *> lhs_view(lhs.data(), basis);
    VectorView<Scalar const*> rhs_view(rhs.data(), basis);

    basic::vector_inplace_addition(
        lhs_view,
        rhs_view
    );

    std::vector<Scalar> expected;
    expected.reserve(basis.size());
    for_each_word(basis,
                  [&expected, &to_idx](auto const& word) {
                      auto idx = to_idx(word);
                      expected.emplace_back(make_scalar({
                              {
                                      {{'a', idx}},
                                      1, 1
                              },
                              {
                                      {{'b', idx}},
                                      1, 1
                              }
                      }));
                  });

    EXPECT_EQ(lhs, expected);
}


TEST_F(VectorInplaceAdditionTests, VectorAdditionScaledRhs)
{
    auto basis = get_basis();

    auto to_idx = word_to_idx_fn(basis);

    auto lhs = make_tensor('a', basis, to_idx);
    const auto rhs = make_tensor('b', basis, to_idx);



    VectorView<Scalar*> lhs_view(lhs.data(), basis);
    VectorView<Scalar const*> rhs_view(rhs.data(), basis);

    auto multiplier = make_scalar({{{{'c', 0}}, 1, 1}});
    basic::vector_inplace_addition(
                           lhs_view,
                           rhs_view,
                           ops::MultiplyBy<Scalar>(multiplier));

    std::vector<Scalar> expected;
    expected.reserve(basis.size());
    for_each_word(basis,
                  [&expected, &to_idx](auto const& word) {
                      auto idx = to_idx(word);
                      expected.emplace_back(make_scalar({
                              {
                                      {{'a', idx}},
                                      1, 1
                              },
                              {
                                      {{'c', 0}, {'b', idx}},
                                      1, 1
                              }
                      }));
                  });

    EXPECT_EQ(lhs, expected);
}

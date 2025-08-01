
#include <gtest/gtest.h>


#include <libalgebra_lite/polynomial.h>
#include <libalgebra_lite/coefficients.h>

#include "roughpy_compute/common/basis.hpp"
#include "roughpy_compute/common/operations.hpp"
#include "roughpy_compute/dense/views.hpp"

#include "vector_addition.hpp"
#include "home/sammorley/CLionProjects/RoughPy/roughpy_compute/dense/views.hpp"


using namespace rpy::compute;

class VectorAdditionTests : public ::testing::Test
{
    std::vector<std::size_t> tensor_begins;
protected:

    using Scalar = typename lal::polynomial_ring::scalar_type;

    using Basis = TensorBasis<>;

    template <typename S>
    using VectorView = DenseVectorView<S*, Basis>;

    static constexpr int32_t width = 3;
    static constexpr int32_t depth = 3;

    void SetUp() override
    {
        tensor_begins.resize(depth+2);

        for (int32_t i=1; i <= depth+1; ++i) {
            tensor_begins[i] = 1 + width * tensor_begins[i-1];
        }
    }


    Basis get_basis() const noexcept
    {
        return { tensor_begins.data(), width, depth };
    }




};





TEST_F(VectorAdditionTests, VectorAdditionIdentityOps)
{
    auto basis = get_basis();
    std::vector<Scalar> lhs;
    std::vector<Scalar> rhs;
    std::vector<Scalar> result;


    VectorView<Scalar const*> lhs_view(lhs.data(), basis);
    VectorView<Scalar const*> rhs_view(rhs.data(), basis);
    VectorView<Scalar*> result_view(result.data(), basis);


    basic::vector_addition(result_view, lhs_view, rhs_view);


}
//
// Created by sammorley on 22/11/24.
//


#include <gtest/gtest.h>

#include <roughpy/core/ranges.h>

#include <roughpy/scalars/scalar_types.h>
#include <roughpy/scalars/scalar_type.h>

#include <roughpy/algebra/free_tensor.h>
#include <roughpy/algebra/context.h>



using namespace rpy;
using namespace rpy::algebra;


namespace {

class FreeTensorTests : public ::testing::Test {
    deg_t width = 2;
    deg_t depth = 5;
    const scalars::ScalarType* rational_tp;
    context_pointer context;

protected:
    void SetUp() override;

public:



    RPY_NO_DISCARD
    FreeTensor generic_tensor(char indeterminate_char) const;

};

void FreeTensorTests::SetUp()
{
    auto tpo = scalars::ScalarType::of<devices::rational_poly_scalar>();
    if (!tpo) {
        GTEST_FAIL();
    }
    rational_tp = *tpo;
    context = get_context(width, depth,  rational_tp);
}

FreeTensor FreeTensorTests::generic_tensor(char indeterminate_char) const
{
    const auto size = context->tensor_size(depth);


    VectorConstructionData cons_data{
        scalars::KeyScalarArray(rational_tp),
        VectorType::Dense
    };

    cons_data.data.allocate_scalars(static_cast<idimn_t>(size));

    auto slice = cons_data.data.as_mut_slice<scalars::rational_poly_scalar>();
    for (auto&& [i, elt] : views::enumerate(slice)) {
        scalars::indeterminate_type ind(indeterminate_char, i);
        elt = scalars::rational_poly_scalar(ind, scalars::rational_scalar_type(1));
    }

    return context->construct_free_tensor(cons_data);
}

}


TEST_F(FreeTensorTests, TestStreamOut)
{
    auto tensor = generic_tensor('x');
    std::stringstream ss;
    ss << tensor;

    const string expected = "{ { 1(x0) }() { 1(x1) }(1)";
    auto first_term = ss.str().substr(0, expected.size());

    EXPECT_EQ(expected, first_term);
}
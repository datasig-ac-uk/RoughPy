//
// Created by sammorley on 09/12/24.
//


#include <gtest/gtest.h>

#include "roughpy/streams/brownian_stream.h"
#include "roughpy/streams/value_stream.h"


using namespace rpy;
using namespace rpy::streams;


namespace {


class TestTensorValuedStream : public ::testing::Test
{
public:
    algebra::context_pointer ctx;
    algebra::FreeTensor initial_condition;
    std::shared_ptr<const StreamInterface> increment_stream;
    std::shared_ptr<const ValueStream<algebra::FreeTensor>> stream;

protected:
    void SetUp() override
    {
        const auto stype = *scalars::ScalarType::of<double>();
        ctx = algebra::get_context(4, 2, stype);

        algebra::VectorConstructionData cdata{
                scalars::KeyScalarArray(stype),
                algebra::VectorType::Dense
        };

        initial_condition = ctx->construct_free_tensor(cdata);

        scalars::seed_int_t seed = 12345;

        StreamMetadata meta{
            4,
            intervals::RealInterval(0.0, 1.0),
            algebra::get_context(4, 3, stype),
            stype,
            algebra::VectorType::Dense,
            4,
            intervals::IntervalType::Clopen
        };

        increment_stream = std::make_shared<BrownianStream>(
            stype->get_rng("", seed),
            std::move(meta));

        stream = make_simple_tensor_valued_stream(
            increment_stream,
            initial_condition,
            intervals::RealInterval(0.0, 1.0)
        );
    }
};

}



TEST_F(TestTensorValuedStream, TestBasicProperties)
{
    EXPECT_EQ(stream->domain(), intervals::RealInterval(0.0, 1.0));
    EXPECT_EQ(stream->initial_value(), initial_condition);
}


TEST_F(TestTensorValuedStream, TestterminalValueSizeCorrect)
{
    EXPECT_EQ(stream->terminal_value().size(), initial_condition.size());
}


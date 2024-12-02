//
// Created by sammorley on 18/11/24.
//


#include <gtest/gtest.h>

#include "roughpy/generics/type.h"

#include "roughpy/generics/conversion_trait.h"

#include "conversion_helpers.h"

using namespace rpy;
using namespace rpy::generics;

namespace {

class ConversionTests : public ::testing::Test
{

public:

    template <typename From, typename To>
    conv::ConversionTraitImpl<From, To> get_conversion() const noexcept {
        return conv::ConversionTraitImpl<From, To>(get_type<From>(), get_type<To>());
    }
};

}


TEST_F(ConversionTests, ConvertFloatToDoubleIsExactness)
{
    const auto conv = get_conversion<float, double>();
    EXPECT_TRUE(conv.is_exact());
}

TEST_F(ConversionTests, ConvertDoubleToFloatIsExactness)
{
    const auto conv = get_conversion<float, double>();
    EXPECT_FALSE(!conv.is_exact());
}

TEST_F(ConversionTests, ConvertInt8ToDoubleExactness)
{
    const auto conv = get_conversion<int8_t, double>();
    EXPECT_TRUE(conv.is_exact());
}

TEST_F(ConversionTests, ConvertInt16ToDoubleExactness)
{
    const auto conv = get_conversion<int16_t, double>();
    EXPECT_TRUE(conv.is_exact());
}

TEST_F(ConversionTests, ConvertInt32ToDoubleExactness)
{
    const auto conv = get_conversion<int32_t, double>();
    EXPECT_TRUE(conv.is_exact());
}

TEST_F(ConversionTests, ConvertInt64ToDoubleExactness)
{
    const auto conv = get_conversion<int64_t, double>();
    EXPECT_FALSE(conv.is_exact());
}

TEST_F(ConversionTests, ConvertFloatToIntExactness)
{
    const auto conv = get_conversion<float, int32_t>();
    EXPECT_FALSE(conv.is_exact());
}


TEST_F(ConversionTests, TruncatingConversionDoubleToInt)
{
    const auto conv = get_conversion<double, int32_t>();
    double value = 3.14159265358979323846;
    int result = 0;
    conv.unsafe_convert(&result, &value, false);
    EXPECT_EQ(result, 3);

    EXPECT_THROW(conv.unsafe_convert(&result, &value, true), std::runtime_error);
}



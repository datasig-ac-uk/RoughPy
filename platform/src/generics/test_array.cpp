#include <gtest/gtest.h>

#include "roughpy/generics/array.h"

using namespace rpy;
using namespace rpy::generics;

TEST(TestArray, TestDefaultConstructor)
{
    Array a;

    ASSERT_EQ(a.type(), nullptr);
    ASSERT_EQ(a.size(), 0);
    ASSERT_EQ(a.capacity(), 0);

    // Operator access on null type should throw
    EXPECT_THROW((void)a[0], ArrayTypeException);
    EXPECT_THROW((void)*a[0].data<float>(), ArrayTypeException);
}

TEST(TestArray, TestTypedContructorZeroSize)
{
    Array a{get_type<float>()};

    ASSERT_EQ(a.type(), get_type<float>());
    ASSERT_EQ(a.size(), 0);
    ASSERT_EQ(a.capacity(), 0);
    ASSERT_TRUE(a.empty());
}

TEST(TestArray, TestTypedContructorWithSize)
{
    Array a{get_type<float>(), 3};

    ASSERT_EQ(a.size(), 3);
    ASSERT_EQ(a.capacity(), 3);
    ASSERT_FALSE(a.empty());

    // FIXME for review - valid alignment checks?
    ASSERT_EQ(reinterpret_cast<std::size_t>(a.data()) % 16, 0);
}

TEST(TestArray, TestTypedContructorWithSizeAligned)
{
    Array a{get_type<float>(), 5, 1};

    ASSERT_EQ(a.size(), 5);
    ASSERT_EQ(a.capacity(), 5);
    ASSERT_FALSE(a.empty());
}

TEST(TestArray, TestCopyConstructAndAssign)
{
    // Original data
    Array a{get_type<double>(), 7};
    for (size_t i = 0; i < 7; ++i) {
        double *v = a[i].data<double>();
        *v = 1.0 / static_cast<double>(i + 1);
    }
    EXPECT_EQ(a.size(), 7);
    EXPECT_EQ(a.capacity(), 7);

    // Copy construct
    Array b(a);
    ASSERT_EQ(b.type(), get_type<double>());
    ASSERT_EQ(b.size(), 7);
    ASSERT_EQ(b.capacity(), 7);

    // Assignment operator
    Array c = a;
    ASSERT_EQ(c.type(), get_type<double>());
    ASSERT_EQ(c.size(), 7);
    ASSERT_EQ(c.capacity(), 7);

    // Confirm values are copied correctly to unique locations
    ASSERT_NE(a.data(), b.data());
    ASSERT_NE(a.data(), c.data());
    for (size_t i = 0; i < 7; ++i) {
        ASSERT_EQ(a[i], b[i]);
        ASSERT_EQ(a[i], c[i]);
    }
}

TEST(TestArray, TestMoveConstructAndAssign)
{
    // Original data
    Array a{get_type<double>(), 11};
    for (size_t i = 0; i < 11; ++i) {
        double *v = a[i].data<double>();
        *v = static_cast<double>(i * i);
    }
    EXPECT_EQ(a.size(), 11);
    EXPECT_EQ(a.capacity(), 11);
    const void* original_data = a.data();

    // Move construct
    Array b(std::move(a));
    ASSERT_EQ(b.type(), get_type<double>());
    ASSERT_EQ(b.size(), 11);
    ASSERT_EQ(b.capacity(), 11);
    ASSERT_EQ(b.data(), original_data); // Data moved
    ASSERT_EQ(a.data(), nullptr); // Old data invalidated

    // Move assignment
    Array c = std::move(b);
    ASSERT_EQ(c.type(), get_type<double>());
    ASSERT_EQ(c.size(), 11);
    ASSERT_EQ(c.capacity(), 11);
    ASSERT_EQ(c.data(), original_data); // Data moved
    ASSERT_EQ(b.data(), nullptr); // Old data invalidated
}

TEST(TestArray, TestArrayAccessor)
{
    Array a{get_type<int>(), 13};
    for (size_t i = 0; i < 13; ++i) {
        int *v = a[i].data<int>();
        *v = static_cast<int>(i);
    }

    // Check const ref and ref in range
    auto const a_const_ref = a[3];
    ASSERT_EQ(*a_const_ref.data<int>(), 3);

    *a[3].data<int>() = -1;
    ASSERT_EQ(*a_const_ref.data<int>(), -1); // Original value reassigned

    // Check const ref and ref outside range
    EXPECT_THROW((void)a[14], ArrayIndexException);
    EXPECT_THROW((void)*a[14].data<int>(), ArrayIndexException);

    // Check optional const ref and ref in range
    auto const a_opt_const_ref = a.get(5);
    ASSERT_EQ(*a_opt_const_ref.value().data<int>(), 5);

    *a.get_mut(5).value().data<int>() = -2;
    ASSERT_EQ(*a_opt_const_ref.value().data<int>(), -2); // Original value reassigned

    // Check optional const ref and ref outside range
    ASSERT_FALSE(a.get(15).has_value());
    ASSERT_FALSE(a.get_mut(15).has_value());

    // Check direct const ref and ref in range
    const auto a_unchecked = a.get_unchecked(7);
    ASSERT_EQ(*a_unchecked.data<int>(), 7);

    *a.get_unchecked_mut(7).data<int>() = -3;
    ASSERT_EQ(*a_unchecked.data<int>(), -3);
}

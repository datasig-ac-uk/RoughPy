#include <gtest/gtest.h>

#include "roughpy/generics/array.h"
#include "roughpy/generics/mocking/mock_type.h"

using namespace rpy;
using namespace rpy::generics;
using namespace rpy::mem;

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

TEST(TestArray, TestDestructorAllocDealloc)
{
    using ::testing::_;
    using namespace rpy::generics::mocking;

    const dimn_t SIZE = 5;
    auto type_ptr = Rc<MockType>{new MockType};
    {
        // Mock double size and confirm data assigned on construction
        EXPECT_CALL(*type_ptr, object_size()).Times(1).WillOnce([]{
            // Arbitrary type size for valid example
            return sizeof(double);
        });
        EXPECT_CALL(*type_ptr, copy_or_fill(_, nullptr, SIZE, true)).Times(1);
        Array arr{type_ptr, SIZE};

        // Data destroyed when arr goes out of scope
        void* data_ptr = arr.data();
        EXPECT_CALL(*type_ptr, destroy_range(data_ptr, SIZE)).Times(1);
    }
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

    EXPECT_TRUE(mem::is_pointer_aligned(a.data(), alignof(std::max_align_t)));

    ASSERT_EQ(*a[0].data<float>(), 0.0f);
    ASSERT_EQ(*a[1].data<float>(), 0.0f);
    ASSERT_EQ(*a[2].data<float>(), 0.0f);
}

TEST(TestArray, TestTypedContructorWithSizeAligned)
{
    for (size_t align : { 16, 32, 64, 128 }) {
        Array a{get_type<uint8_t>(), 5, align};
        EXPECT_TRUE(mem::is_pointer_aligned(a.data(), align));
    }
}

TEST(TestArray, TestCopyConstructAndAssign)
{
    // Original data
    Array a{get_type<double>(), 7};
    for (std::size_t i = 0; i < 7; ++i) {
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
    for (std::size_t i = 0; i < 7; ++i) {
        ASSERT_EQ(a[i], b[i]);
        ASSERT_EQ(a[i], c[i]);
    }
}

TEST(TestArray, TestMoveConstructAndAssign)
{
    // Original data
    Array a{get_type<double>(), 11};
    for (std::size_t i = 0; i < 11; ++i) {
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
    for (std::size_t i = 0; i < 13; ++i) {
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

TEST(TestArray, TestArrayReserveNoType)
{
    Array a{};
    EXPECT_THROW(a.reserve(10), ArrayTypeException);
}

TEST(TestArray, TestArrayResizeNoType)
{
    Array a{};
    EXPECT_THROW(a.resize(10), ArrayTypeException);
}

TEST(TestArray, TestArrayReserve)
{
    Array a{get_type<int16_t>(), 17};
    for (std::size_t i = 0; i < 17; ++i) {
        *a[i].data<int16_t>() = static_cast<int16_t>(i);
    }

    const void* old_data = a.data();
    a.reserve(30);

    // Confirm data has been relocated and copied correctly
    ASSERT_EQ(a.size(), 17);
    ASSERT_EQ(a.capacity(), 30);
    ASSERT_NE(a.data(), old_data);
    for (std::size_t i = 0; i < 17; ++i) {
        ASSERT_EQ(*a[i].data<int16_t>(), static_cast<int16_t>(i));
    }
}

TEST(TestArray, TestArrayResize)
{
    Array a{get_type<uint16_t>(), 19};
    for (std::size_t i = 0; i < 19; ++i) {
        *a[i].data<uint16_t>() = static_cast<uint16_t>(i);
    }

    const void* old_data = a.data();
    a.resize(40);

    // Confirm relocation and copy has happened as in reserve test above
    ASSERT_EQ(a.size(), 40);
    ASSERT_EQ(a.capacity(), 40);
    ASSERT_NE(a.data(), old_data);

    // Old values must copied and new ones must be default constructed
    for (std::size_t i = 0; i < 40; ++i) {
        if (i < 19) {
            ASSERT_EQ(*a[i].data<uint16_t>(), static_cast<uint16_t>(i));
        } else {
            ASSERT_EQ(*a[i].data<uint16_t>(), 0);
        }
    }
}

TEST(TestArray, TestArrayResizeNoRelocate)
{
    Array a{get_type<uint8_t>(), 7};
    for (std::size_t i = 0; i < 7; ++i) {
        *a[i].data<uint8_t>() = static_cast<uint8_t>(i);
    }

    // Reserve should realloc
    const void* old_data1 = a.data();
    a.reserve(20);

    // Expect relocation of data as above
    EXPECT_EQ(a.size(), 7);
    EXPECT_EQ(a.capacity(), 20);
    EXPECT_NE(a.data(), old_data1);
    for (std::size_t i = 0; i < 7; ++i) {
        EXPECT_EQ(*a[i].data<uint8_t>(), static_cast<uint8_t>(i));
    }

    // Reserve under capacity will preserve old data
    const void* old_data2 = a.data();
    a.resize(13);

    // Resized but new data still at old ptr
    ASSERT_EQ(a.size(), 13);
    ASSERT_EQ(a.capacity(), 20);
    ASSERT_EQ(a.data(), old_data2);

    // Old values must copied and new ones must be default constructed
    for (std::size_t i = 0; i < 13; ++i) {
        if (i < 7) {
            ASSERT_EQ(*a[i].data<uint8_t>(), static_cast<uint8_t>(i));
        } else {
            ASSERT_EQ(*a[i].data<uint8_t>(), 0);
        }
    }
}

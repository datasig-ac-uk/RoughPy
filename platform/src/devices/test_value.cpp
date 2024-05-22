//
// Created by sam on 5/21/24.
//

#include <roughpy/devices/value.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <roughpy/core/container/vector.h>
#include <roughpy/core/ranges.h>
#include <roughpy/devices/buffer.h>
#include <roughpy/devices/type.h>

using namespace rpy;
using namespace rpy::devices;

namespace {

class TestValue : public testing::Test
{
protected:
    TestValue() : double_type(get_type("f64")) {}

    const Type* double_type;
};

}// namespace

TEST_F(TestValue, TestAddInplace)
{
    Value left(0.5);
    Value right(1.53);

    left += right;
    EXPECT_EQ(left, Value(0.5 + 1.53));
}

TEST_F(TestValue, TestSubtractInplace)
{
    Value left(3.0);
    Value right(1.5);

    left -= right;
    EXPECT_EQ(left, Value(3.0 - 1.5));
}

TEST_F(TestValue, TestMultiplyInplace)
{
    Value left(7.0);
    Value right(2.0);

    left *= right;
    EXPECT_EQ(left, Value(7.0 * 2.0));
}

TEST_F(TestValue, TestDivideInplace)
{
    Value left(10.0);
    Value right(2.0);

    left /= right;
    EXPECT_EQ(left, Value(10.0 / 2.0));
}

TEST_F(TestValue, TestAdd)
{
    Value left(5.0);
    Value right(10.0);

    Value result = left + right;
    EXPECT_EQ(result, Value(5.0 + 10.0));
}

TEST_F(TestValue, TestSubtract)
{
    Value left(20.0);
    Value right(5.0);

    Value result = left - right;
    EXPECT_EQ(result, Value(20.0 - 5.0));
}

TEST_F(TestValue, TestMultiply)
{
    Value left(3.0);
    Value right(6.0);

    Value result = left * right;
    EXPECT_EQ(result, Value(3.0 * 6.0));
}

TEST_F(TestValue, TestDivide)
{
    Value left(12.0);
    Value right(4.0);

    Value result = left / right;
    EXPECT_EQ(result, Value(12.0 / 4.0));
}

TEST_F(TestValue, TestChangeType)
{
    Value value(15.0);
    ASSERT_EQ(value.type(), double_type);

    const auto* i32_type = get_type("i32");
    value.change_type(i32_type);
    EXPECT_EQ(value.type(), i32_type);
    EXPECT_EQ(value, Value(15));
}

TEST_F(TestValue, TestAssignFromInt)
{
    Value value(double_type);

    value = 15;

    EXPECT_EQ(value.type(), double_type);
    EXPECT_EQ(value, Value(15.0));
}

TEST_F(TestValue, AssignIntFromFloat)
{
    Value value(get_type("i32"));

    value = 15.5;

    EXPECT_EQ(value, Value(15));
}

TEST_F(TestValue, AssignFloatToEmpty)
{
    Value value;
    value = 15.5;

    EXPECT_EQ(value.type(), double_type);
    EXPECT_EQ(value, Value(15.5));
}

TEST_F(TestValue, TestLessThan)
{
    Value smaller(10.0);
    Value larger(20.0);
    EXPECT_TRUE(smaller < larger);
}

TEST_F(TestValue, TestLessThanOrEqual)
{
    Value smallerOrEqual1(10.0);
    Value smallerOrEqual2(10.0);
    Value larger(20.0);
    EXPECT_TRUE(smallerOrEqual1 <= larger);
    EXPECT_TRUE(smallerOrEqual2 <= smallerOrEqual2);
}

TEST_F(TestValue, TestGreaterThan)
{
    Value larger(20.0);
    Value smaller(10.0);
    EXPECT_TRUE(larger > smaller);
}

TEST_F(TestValue, TestGreaterThanOrEqual)
{
    Value largerOrEqual1(20.0);
    Value largerOrEqual2(20.0);
    Value smaller(10.0);
    EXPECT_TRUE(largerOrEqual1 >= smaller);
    EXPECT_TRUE(largerOrEqual2 >= largerOrEqual2);
}

TEST_F(TestValue, TestEqual)
{
    Value val1(10.0);
    Value val2(10.0);
    EXPECT_TRUE(val1 == val2);
}

TEST_F(TestValue, TestNotEqual)
{
    Value val1(10.0);
    Value val2(20.0);
    EXPECT_TRUE(val1 != val2);
}

TEST_F(TestValue, TestStreamOutOperator)
{
    Value testValue(5.5);
    std::ostringstream oss;

    oss << testValue;

    EXPECT_EQ(oss.str(), "5.5");
}

TEST_F(TestValue, TestAddInplaceInteger)
{
    Value left(0.5);
    int right = 1;

    left += right;
    EXPECT_EQ(left, Value(0.5 + 1));
}

TEST_F(TestValue, TestSubtractInplaceInteger)
{
    Value left(3.0);
    int right = 1;

    left -= right;
    EXPECT_EQ(left, Value(3.0 - 1));
}

TEST_F(TestValue, TestMultiplyInplaceInteger)
{
    Value left(7.0);
    int right = 2;

    left *= right;
    EXPECT_EQ(left, Value(7.0 * 2));
}

TEST_F(TestValue, TestDivideInplaceInteger)
{
    Value left(10.0);
    int right = 2;

    left /= right;
    EXPECT_EQ(left, Value(10.0 / 2));
}

namespace {

class MockType : public Type
{
public:
    struct Val {
        double value;
        int id;
        bool in_use;
    };

private:
    mutable containers::Vec<Val> m_values;

public:
    MockType()
        : Type("mock",
               "mock",
               {TypeCode::OpaqueHandle, sizeof(void*), alignof(void*), 1},
               TypeTraits{})
    {
        m_values.reserve(10);
        for (int i = 0; i < 10; ++i) {
            m_values.emplace_back(Val{0.0, i, false});
        }

        {
            auto val_support = update_support(this);
            val_support->arithmetic.add_inplace
                    = [](void* dst, const void* src) {
                          static_cast<Val*>(dst)->value
                                  += static_cast<const Val*>(src)->value;
                      };
            val_support->arithmetic.sub_inplace
                    = [](void* dst, const void* src) {
                          static_cast<Val*>(dst)->value
                                  -= static_cast<const Val*>(src)->value;
                      };
            val_support->arithmetic.mul_inplace
                    = [](void* dst, const void* src) {
                          static_cast<Val*>(dst)->value
                                  *= static_cast<const Val*>(src)->value;
                      };
            val_support->arithmetic.div_inplace
                    = [](void* dst, const void* src) {
                          static_cast<Val*>(dst)->value
                                  /= static_cast<const Val*>(src)->value;
                      };

            // Comparison support functions
            val_support->comparison.equals
                    = [](const void* val, const void* dbl) {
                          return static_cast<const Val*>(val)->value
                                  == static_cast<const Val*>(dbl)->value;
                      };
            val_support->comparison.less
                    = [](const void* val, const void* dbl) {
                          return static_cast<const Val*>(val)->value
                                  < static_cast<const Val*>(dbl)->value;
                      };
            val_support->comparison.less_equal
                    = [](const void* val, const void* dbl) {
                          return static_cast<const Val*>(val)->value
                                  <= static_cast<const Val*>(dbl)->value;
                      };
            val_support->comparison.greater
                    = [](const void* val, const void* dbl) {
                          return static_cast<const Val*>(val)->value
                                  > static_cast<const Val*>(dbl)->value;
                      };
            val_support->comparison.greater_equal
                    = [](const void* val, const void* dbl) {
                          return static_cast<const Val*>(val)->value
                                  >= static_cast<const Val*>(dbl)->value;
                      };

            // Type conversion support functions
            val_support->conversions.convert = [](void* dst, const void* src) {
                static_cast<Val*>(dst)->value
                        = static_cast<const Val*>(src)->value;
            };
            val_support->conversions.move_convert = [](void* dst, void* src) {
                static_cast<Val*>(dst)->value = static_cast<Val*>(src)->value;
                static_cast<Val*>(src)->value
                        = 0;// Moving therefore source is reset
            };
        }
    }

    const containers::Vec<Val>& vals() const noexcept { return m_values; }

    template <typename T>
    void add_support() const;

    void* allocate_single() const override;
    void free_single(void* ptr) const override;

    void copy(void* dst, const void* src, dimn_t count) const override;
    void move(void* dst, void* src, dimn_t count) const override;
    void display(std::ostream& os, const void* ptr) const override;
};

void* MockType::allocate_single() const
{
    auto v = rpy::ranges::find_if(m_values, [](const Val& val) {
        return !val.in_use;
    });
    RPY_CHECK(v != m_values.end());
    v->in_use = true;
    return &*v;
}

void MockType::free_single(void* ptr) const
{
    auto* val = static_cast<Val*>(ptr);
    val->in_use = false;
}

void MockType::copy(void* dst, const void* src, dimn_t count) const
{
    auto src_val = static_cast<const Val*>(src);
    auto dst_val = static_cast<Val*>(dst);

    for (int i = 0; i < count; ++i) { dst_val[i].value = src_val[i].value; }
}

void MockType::move(void* dst, void* src, dimn_t count) const
{
    auto src_val = static_cast<Val*>(src);
    auto dst_val = static_cast<Val*>(dst);

    for (int i = 0; i < count; ++i) {
        dst_val[i].value = src_val[i].value;

        src_val[i].in_use = false;
        src_val[i].value = 0.0;
    }
}

void MockType::display(std::ostream& os, const void* ptr) const
{
    auto val_ptr = static_cast<const Val*>(ptr);

    os << val_ptr->id << " " << val_ptr->value;
}

template <typename T>
void MockType::add_support() const
{
    auto support = this->update_support(get_type<T>());

    support->arithmetic.add_inplace = [](void* dst, const void* src) {
        static_cast<Val*>(dst)->value += *static_cast<const T*>(src);
    };
    support->arithmetic.sub_inplace = [](void* dst, const void* src) {
        static_cast<Val*>(dst)->value -= *static_cast<const T*>(src);
    };
    support->arithmetic.mul_inplace = [](void* dst, const void* src) {
        static_cast<Val*>(dst)->value *= *static_cast<const T*>(src);
    };
    support->arithmetic.div_inplace = [](void* dst, const void* src) {
        static_cast<Val*>(dst)->value /= *static_cast<const T*>(src);
    };

    support->comparison.equals = [](const void* val, const void* dbl) {
        return static_cast<const Val*>(val)->value
                == *static_cast<const T*>(dbl);
    };
    support->comparison.less = [](const void* val, const void* dbl) {
        return static_cast<const Val*>(val)->value
                < *static_cast<const T*>(dbl);
    };
    support->comparison.less_equal = [](const void* val, const void* dbl) {
        return static_cast<const Val*>(val)->value
                <= *static_cast<const T*>(dbl);
    };
    support->comparison.greater = [](const void* val, const void* dbl) {
        return static_cast<const Val*>(val)->value
                > *static_cast<const T*>(dbl);
    };
    support->comparison.greater_equal = [](const void* val, const void* dbl) {
        return static_cast<const Val*>(val)->value
                >= *static_cast<const T*>(dbl);
    };

    support->conversions.convert = [](void* dst, const void* src) {
        static_cast<Val*>(dst)->value = *static_cast<const T*>(src);
    };
    support->conversions.move_convert = [](void* dst, void* src) {
        static_cast<Val*>(dst)->value = *static_cast<const T*>(src);
    };
}

class TestMockTypeValue : public ::testing::Test
{
    static const MockType m_type;

protected:
    TestMockTypeValue()
    {
        type = &m_type;
        m_type.add_support<double>();
    }

    double val_of(const Value& value)
    {
        RPY_CHECK(value.type() == type);
        return value.data<typename MockType::Val>()->value;
    }

    int id_of(const Value& value)
    {
        RPY_CHECK(value.type() == type);
        return value.data<typename MockType::Val>()->id;
    }

    bool in_use(int id)
    {
        RPY_CHECK(id < m_type.vals().size());
        return m_type.vals()[id].in_use;
    }

    const Type* type;
};

const MockType TestMockTypeValue::m_type{};

}// namespace

TEST_F(TestMockTypeValue, TestConstructValueDouble)
{
    Value value(type, 1.0);
    EXPECT_EQ(val_of(value), 1.0);
}

TEST_F(TestMockTypeValue, TestCopyConstructor)
{
    Value value1(type, 2.0);
    Value value2 = value1;
    EXPECT_EQ(val_of(value1), val_of(value2));
}

TEST_F(TestMockTypeValue, TestMoveConstructor)
{
    Value value1(type, 3.0);
    auto id = id_of(value1);
    Value value2(std::move(value1));
    EXPECT_EQ(val_of(value2), 3.0);
    EXPECT_TRUE(value1.fast_is_zero());
    EXPECT_EQ(id_of(value2), id);
}

TEST_F(TestMockTypeValue, TestConstructionFromIntFails)
{
    EXPECT_THROW(Value value(type, 4), std::runtime_error);
}

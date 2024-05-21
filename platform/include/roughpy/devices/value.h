//
// Created by sam on 3/29/24.
//

// ReSharper disable CppUseStructuredBinding
// ReSharper disable CppTooWideScopeInitStatement
// ReSharper disable CppNonExplicitConvertingConstructor
#ifndef ROUGHPY_DEVICES_VALUE_H
#define ROUGHPY_DEVICES_VALUE_H

#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include "core.h"
#include "type.h"

namespace rpy {
namespace devices {

class ConstReference;
template <typename T>
class TypedConstReference;
class Reference;
template <typename T>
class TypedReference;

class RPY_DEVICES_EXPORT Value
{
    friend class ConstReference;
    friend class Reference;

    const Type* p_type = nullptr;
    union Storage
    {
        constexpr Storage() : pointer(nullptr) {}

        alignas(void*) byte bytes[sizeof(void*)];
        void* pointer;
    };

    Storage m_storage;

    RPY_NO_DISCARD static bool is_inline_stored(const Type* type) noexcept
    {
        return type != nullptr && traits::is_arithmetic(type)
                && size_of(type) <= sizeof(void*);
    }

    RPY_NO_DISCARD bool is_inline_stored() const noexcept
    {
        return is_inline_stored(p_type);
    }

public:
    Value() = default;

    Value(const Value& other);
    Value(Value&& other) noexcept;
    explicit Value(ConstReference other);

    template <
            typename T,
            typename = enable_if_t<
                    !is_same_v<decay_t<T>, Value>
                    || !is_same_v<decay_t<T>, ConstReference>
                    || !is_same_v<decay_t<T>, Reference>>>
    explicit Value(T&& other) noexcept : p_type(get_type<T>())
    {
        T* this_ptr;
        if (is_inline_stored()) {
            this_ptr = m_storage.bytes;
        } else {
            this_ptr = p_type->allocate_single();
        }
        construct_inplace(this_ptr, std::forward<T>(other));
    }

    ~Value();

    RPY_NO_DISCARD const Type* type() const noexcept { return p_type; }

    template <typename T = void>
    RPY_NO_DISCARD const T* data() const noexcept
    {
        if (is_inline_stored()) {
            return static_cast<const T*>(m_storage.bytes);
        }
        return static_cast<const T*>(m_storage.pointer);
    }

    template <typename T = void>
    RPY_NO_DISCARD T* data() noexcept
    {
        if (is_inline_stored()) { return static_cast<T*>(m_storage.bytes); }
        return static_cast<T*>(m_storage.pointer);
    }

    Value& operator=(const Value& other);
    Value& operator=(Value&& other) noexcept;

    template <typename T>
    enable_if_t<
            !(is_same_v<decay_t<T>, Value>
              || is_same_v<decay_t<T>, ConstReference>
              || is_same_v<decay_t<T>, Reference>),
            Value&>
    operator=(T&& other);

    void change_type(const Type* new_type);

    // ReSharper disable CppNonExplicitConversionOperator
    operator ConstReference() const noexcept;// NOLINT(*-explicit-constructor)
    operator Reference() noexcept;           // NOLINT(*-explicit-constructor)
    // ReSharper enable CppNonExplicitConversionOperator

    Value& operator+=(ConstReference other);
    Value& operator-=(ConstReference other);
    Value& operator*=(ConstReference other);
    Value& operator/=(ConstReference other);

    Value& operator+=(const Value& other);
    Value& operator-=(const Value& other);
    Value& operator*=(const Value& other);
    Value& operator/=(const Value& other);

    template <typename T>
    Value& operator+=(const T& other);
    template <typename T>
    Value& operator-=(const T& other);
    template <typename T>
    Value& operator*=(const T& other);
    template <typename T>
    Value& operator/=(const T& other);
};

class ConstReference
{
    const void* p_val;
    const Type* p_type;

public:
    constexpr explicit ConstReference(const void* val, const Type* type)
        : p_val(val),
          p_type(type)
    {
        RPY_CHECK(type != nullptr);
    }

    RPY_NO_DISCARD const Type* type() const noexcept { return p_type; }
    template <typename T = void>
    RPY_NO_DISCARD const T* data() const noexcept
    {
        return p_val;
    }

    template <typename T>
    constexpr explicit operator TypedConstReference<T>() const noexcept
    {
        return TypedConstReference<T>(value<T>());
    }

    template <typename T>
    constexpr add_const_t<T>& value() const
    {
        return *static_cast<add_const_t<T>*>(p_val);
    }
};

template <typename T>
class TypedConstReference : public ConstReference
{
public:
    explicit TypedConstReference(const T& val)
        : ConstReference(std::addressof(val), get_type<T>())
    {}

    template <typename U>
    explicit
    operator enable_if_t<is_convertible_v<T, U>, TypedConstReference<U>>()
    {
        return TypedConstReference(ConstReference::value<U>());
    }

    using ConstReference::value;

    constexpr add_const_t<T>& value() const noexcept
    {
        return ConstReference::value<T>();
    }
};

class Reference : public ConstReference
{
public:
    // ReSharper disable once CppParameterMayBeConstPtrOrRef
    constexpr explicit Reference(void* val, const Type* type)
        : ConstReference(val, type)
    {}

    using ConstReference::data;

    template <typename T>
    enable_if_t<
            !(is_same_v<decay_t<T>, Value> || is_same_v<decay_t<T>, Reference>
              || is_same_v<decay_t<T>, ConstReference>),
            Reference&>
    operator=(T&& other);

    template <typename T = void>
    RPY_NO_DISCARD T* data() noexcept
    {
        return const_cast<T*>(ConstReference::data<T>());
    }

    template <typename T>
    explicit constexpr operator TypedConstReference<T>() noexcept
    {
        return TypedConstReference(value<T>());
    }

    template <typename T>
    explicit constexpr operator TypedReference<T>() noexcept
    {
        return TypedReference(value<T>());
    }

    template <typename T>
    constexpr T& value()
    {
        return const_cast<T&>(ConstReference::value<T>());
    }

    Reference& operator+=(const ConstReference other)
    {
        const auto& arithmetic = type()->arithmetic(other.type());
        RPY_CHECK(arithmetic.add_inplace != nullptr);
        if (type() == other.type()) {
            arithmetic.add_inplace(data(), other.data());
        } else {
            RPY_THROW(std::runtime_error, "Type mismatch error.");
        }

        return *this;
    }

    Reference& operator-=(const ConstReference other)
    {
        const auto& arithmetic = type()->arithmetic(other.type());
        RPY_CHECK(arithmetic.sub_inplace != nullptr);
        if (type() == other.type()) {
            arithmetic.sub_inplace(data(), other.data());
        } else {
            RPY_THROW(std::runtime_error, "Type mismatch error.");
        }

        return *this;
    }

    Reference& operator*=(const ConstReference other)
    {
        const auto& arithmetic = type()->arithmetic(other.type());
        RPY_CHECK(arithmetic.mul_inplace != nullptr);
        if (type() == other.type()) {
            arithmetic.mul_inplace(data(), other.data());
        } else {
            RPY_THROW(std::runtime_error, "Type mismatch error.");
        }

        return *this;
    }

    Reference& operator/=(const ConstReference other)
    {
        const auto& arithmetic = type()->arithmetic(other.type());
        RPY_CHECK(arithmetic.div_inplace != nullptr);
        if (type() == other.type()) {
            arithmetic.div_inplace(data(), other.data());
        } else {
            RPY_THROW(std::runtime_error, "Type mismatch error.");
        }

        return *this;
    }
};

template <typename T>
class TypedReference : public Reference
{
public:
    constexpr TypedReference(T& t)// NOLINT(*-explicit-constructor)
        : Reference(
                  const_cast<remove_cv_t<T>*>(std::addressof(t)),
                  devices::get_type<T>()
          )
    {}

    using Reference::value;

    // ReSharper disable CppNonExplicitConversionOperator
    constexpr operator TypedConstReference<T>(// NOLINT(*-explicit-constructor)
    ) const noexcept
    {
        return TypedConstReference(value<T>());
    }

    constexpr operator add_const_t<T>&() const// NOLINT(*-explicit-constructor)
    {
        return *value<add_const_t<T>>();
    }

    template <typename U = remove_cv_ref_t<T>>
    constexpr operator enable_if_t<// NOLINT(*-explicit-constructor)
            is_convertible_v<T&, U&>,
            U&>()
    {
        return static_cast<U&>(value<T>());
    }
    // ReSharper enable CppNonExplicitConversionOperator
};

inline Value::operator ConstReference() const noexcept
{
    return ConstReference(data(), p_type);
}

inline Value::operator Reference() noexcept
{
    return Reference(data(), p_type);
}

template <typename T>
enable_if_t<
        !(is_same_v<decay_t<T>, Value> || is_same_v<decay_t<T>, ConstReference>
          || is_same_v<decay_t<T>, Reference>),
        Value&>
Value::operator=(T&& other)
{
    if (p_type) {
        // Convert the value of other to the current type
        const auto& conversion = p_type->conversions(get_type<T>());
        if (is_rvalue_reference_v<T> && conversion.move_convert) {
            conversion.move_convert(data(), &other);
        } else {
            conversion.convert(data(), &other);
        }
    } else {
        // Use the type of T to construct this.
        construct_inplace(this, std::forward<T>(other));
    }
    return *this;
}

template <typename T>
enable_if_t<
        !(is_same_v<decay_t<T>, Value> || is_same_v<decay_t<T>, Reference>
          || is_same_v<decay_t<T>, ConstReference>),
        Reference&>
Reference::operator=(T&& other)
{
    RPY_DBG_ASSERT(type() != nullptr);
    const auto* tp = get_type<T>();
    const auto& conversion = type()->conversions(tp);
    if (is_rvalue_reference_v<T> && conversion.move_convert) {
        conversion.move_convert(data(), &other);
    } else {
        conversion.convert(data(), &other);
    }
    return *this;
}

inline Value& Value::operator+=(const ConstReference other)
{
    const auto& arithmetic = p_type->arithmetic(other.type());
    RPY_CHECK(arithmetic.add_inplace != nullptr);
    if (p_type == other.type()) {
        arithmetic.add_inplace(this->data(), other.data());
    } else {
        RPY_THROW(std::runtime_error, "Type mismatch error.");
    }

    return *this;
}

inline Value& Value::operator-=(const ConstReference other)
{
    const auto& arithmetic = p_type->arithmetic(other.type());
    RPY_CHECK(arithmetic.sub_inplace != nullptr);
    if (p_type == other.type()) {
        arithmetic.sub_inplace(this->data(), other.data());
    } else {
        RPY_THROW(std::runtime_error, "Type mismatch error.");
    }

    return *this;
}

inline Value& Value::operator*=(const ConstReference other)
{
    const auto& arithmetic = p_type->arithmetic(other.type());
    RPY_CHECK(arithmetic.mul_inplace != nullptr);
    if (p_type == other.type()) {
        arithmetic.mul_inplace(this->data(), other.data());
    } else {
        RPY_THROW(std::runtime_error, "Type mismatch error.");
    }

    return *this;
}

inline Value& Value::operator/=(const ConstReference other)
{
    const auto& arithmetic = p_type->arithmetic(other.type());
    RPY_CHECK(arithmetic.div_inplace != nullptr);
    if (p_type == other.type()) {
        arithmetic.div_inplace(this->data(), other.data());
    } else {
        RPY_THROW(std::runtime_error, "Type mismatch error.");
    }

    return *this;
}

template <typename T>
Value& Value::operator+=(const T& other)
{
    if (other != T()) { operator+=(ConstReference(&other, get_type<T>())); }
    return *this;
}
template <typename T>
Value& Value::operator-=(const T& other)
{
    if (other != T()) { operator-=(ConstReference(&other, get_type<T>())); }
    return *this;
}
template <typename T>
Value& Value::operator*=(const T& other)
{
    if (other != T()) {
        operator*=(ConstReference(&other, get_type<T>()));
    } else {
        operator=(T());
    }
    return *this;
}
template <typename T>
Value& Value::operator/=(const T& other)
{
    if (other == T()) { RPY_THROW(std::domain_error, "division by zero"); }
    return operator/=(constReference(&other, get_type<T>()));
}

inline Value& Value::operator+=(const Value& other)
{
    if (other.p_type != nullptr) {
        return operator+=(static_cast<ConstReference>(other));
    }
    return *this;
}
inline Value& Value::operator-=(const Value& other)
{
    if (other.p_type != nullptr) {
        return operator-=(static_cast<ConstReference>(other));
    }
    return *this;
}
inline Value& Value::operator*=(const Value& other)
{
    if (other.p_type != nullptr) {
        return operator*=(static_cast<ConstReference>(other));
    }
    return operator=(0);
}
inline Value& Value::operator/=(const Value& other)
{
    if (other.p_type != nullptr) {
        return operator/=(static_cast<ConstReference>(other));
    }

    RPY_THROW(std::domain_error, "division by zero");
}

inline Value operator+(const Value& left, const Value& right)
{
    Value result(left);
    result += right;
    return result;
}

inline Value operator-(const Value& left, const Value& right)
{
    Value result(left);
    result -= right;
    return result;
}

inline Value operator*(const Value& left, const Value& right)
{
    Value result(left);
    result *= right;
    return result;
}

inline Value operator/(const Value& left, const Value& right)
{
    Value result(left);
    result /= right;
    return result;
}

inline Value operator+(const ConstReference& left, const ConstReference& right)
{
    Value result(left);
    result += right;
    return result;
}

inline Value operator-(const ConstReference& left, const ConstReference& right)
{
    Value result(left);
    result -= right;
    return result;
}

inline Value operator*(const ConstReference& left, const ConstReference& right)
{
    Value result(left);
    result *= right;
    return result;
}

inline Value operator/(const ConstReference& left, const ConstReference& right)
{
    Value result(left);
    result /= right;
    return result;
}

RPY_NO_DISCARD inline bool operator==(const Value& left, const Value& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.equals);
    return comparisons.equals(left.data(), right.data());
}

// Inequality operator
RPY_NO_DISCARD inline bool operator!=(const Value& left, const Value& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.equals);
    return !comparisons.equals(left.data(), right.data());
}

// Greater than operator
RPY_NO_DISCARD inline bool operator>(const Value& left, const Value& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.greater);
    return comparisons.greater(left.data(), right.data());
}

// Less than operator
RPY_NO_DISCARD inline bool operator<(const Value& left, const Value& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.less);
    return comparisons.less(left.data(), right.data());
}

// Greater than or equal operator
RPY_NO_DISCARD inline bool operator>=(const Value& left, const Value& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.greater_equal);
    return comparisons.greater_equal(left.data(), right.data());
}

// Less than or equal operator
RPY_NO_DISCARD inline bool operator<=(const Value& left, const Value& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.less_equal);
    return comparisons.less_equal(left.data(), right.data());
}

// Equality operator
inline bool operator==(const ConstReference& left, const ConstReference& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.equals);
    return comparisons.equals(left.data(), right.data());
}

// Inequality operator
inline bool operator!=(const ConstReference& left, const ConstReference& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.equals);
    return !comparisons.equals(left.data(), right.data());
}

// Greater than operator
inline bool operator>(const ConstReference& left, const ConstReference& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.greater);
    return comparisons.greater(left.data(), right.data());
}

// Less than operator
inline bool operator<(const ConstReference& left, const ConstReference& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.less);
    return comparisons.less(left.data(), right.data());
}

// Greater than or equal operator
inline bool operator>=(const ConstReference& left, const ConstReference& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.greater_equal);
    return comparisons.greater_equal(left.data(), right.data());
}

// Less than or equal operator
inline bool operator<=(const ConstReference& left, const ConstReference& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.less_equal);
    return comparisons.less_equal(left.data(), right.data());
}

// Equality operator
inline bool operator==(const Value& left, const ConstReference& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.equals);
    return comparisons.equals(left.data(), right.data());
}

// Inequality operator
inline bool operator!=(const Value& left, const ConstReference& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.equals);
    return !comparisons.equals(left.data(), right.data());
}

// Greater than operator
inline bool operator>(const Value& left, const ConstReference& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.greater);
    return comparisons.greater(left.data(), right.data());
}

// Less than operator
inline bool operator<(const Value& left, const ConstReference& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.less);
    return comparisons.less(left.data(), right.data());
}

// Greater than or equal operator
inline bool operator>=(const Value& left, const ConstReference& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.greater_equal);
    return comparisons.greater_equal(left.data(), right.data());
}

// Less than or equal operator
inline bool operator<=(const Value& left, const ConstReference& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.less_equal);
    return comparisons.less_equal(left.data(), right.data());
}

// Equality operator
inline bool operator==(const ConstReference& left, const Value& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.equals);
    return comparisons.equals(left.data(), right.data());
}

// Inequality operator
inline bool operator!=(const ConstReference& left, const Value& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.equals);
    return !comparisons.equals(left.data(), right.data());
}

// Greater than operator
inline bool operator>(const ConstReference& left, const Value& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.greater);
    return comparisons.greater(left.data(), right.data());
}

// Less than operator
inline bool operator<(const ConstReference& left, const Value& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.less);
    return comparisons.less(left.data(), right.data());
}

// Greater than or equal operator
inline bool operator>=(const ConstReference& left, const Value& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.greater_equal);
    return comparisons.greater_equal(left.data(), right.data());
}

// Less than or equal operator
inline bool operator<=(const ConstReference& left, const Value& right)
{
    auto& comparisons = left.type()->comparisons(right.type());
    RPY_CHECK(comparisons.less_equal);
    return comparisons.less_equal(left.data(), right.data());
}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_VALUE_H

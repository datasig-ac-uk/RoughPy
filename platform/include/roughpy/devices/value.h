//
// Created by sam on 3/29/24.
//

#ifndef ROUGHPY_DEVICES_VALUE_H
#define ROUGHPY_DEVICES_VALUE_H

#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include "roughpy/core/strings.h"
#include "type.h"

namespace rpy {
namespace devices {

class ConstReference;
template <typename T>
class TypedConstReference;
class Reference;
template <typename T>
class TypedReference;

class Value
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

    RPY_NO_DISCARD bool is_inline_stored() const noexcept
    {
        return p_type != nullptr && traits::is_arithmetic(p_type)
                && size_of(p_type) <= sizeof(void*);
    }

public:
    Value() = default;

    Value(const Value& other);
    Value(Value&& other) noexcept;
    explicit Value(ConstReference other);

    template <
            typename T,
            typename = enable_if_t<
                    !is_same_v<T, Value> || !is_same_v<T, ConstReference>
                    || !is_same_v<T, Reference>>>
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

    // ReSharper disable CppNonExplicitConversionOperator
    operator ConstReference() const noexcept;// NOLINT(*-explicit-constructor)
    operator Reference() noexcept;           // NOLINT(*-explicit-constructor)
    // ReSharper enable CppNonExplicitConversionOperator

    Value& operator+=(ConstReference other);
    Value& operator-=(ConstReference other);
    Value& operator*=(ConstReference other);
    Value& operator/=(ConstReference other);
};

class ConstReference
{
    const void* p_val;
    const Type* p_type;

public:
    constexpr explicit ConstReference(const void* val, const Type* type)
        : p_val(val),
          p_type(type)
    {}

    RPY_NO_DISCARD const Type* type() const noexcept { return p_type; }
    RPY_NO_DISCARD const void* data() const noexcept { return p_val; }

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
        : ConstReference(std::addressof(val))
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
    constexpr explicit Reference(void* val, const Type* type)
        : ConstReference(val, type)
    {}

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

    Reference& operator+=(ConstReference other)
    {
        const auto& arithmetic = type()->arithmetic(other.type());
        RPY_CHECK(arithmetic.add_inplace != nullptr);
        if (type() == other.type()) {
            arithmetic.add_inplace(const_cast<void*>(data()), other.data());
        } else {
            RPY_THROW(std::runtime_error, "Type mismatch error.");
        }

        return *this;
    }

    Reference& operator-=(ConstReference other)
    {
        const auto& arithmetic = type()->arithmetic(other.type());
        RPY_CHECK(arithmetic.sub_inplace != nullptr);
        if (type() == other.type()) {
            arithmetic.sub_inplace(const_cast<void*>(data()), other.data());
        } else {
            RPY_THROW(std::runtime_error, "Type mismatch error.");
        }

        return *this;
    }

    Reference& operator*=(ConstReference other)
    {
        const auto& arithmetic = type()->arithmetic(other.type());
        RPY_CHECK(arithmetic.mul_inplace != nullptr);
        if (type() == other.type()) {
            arithmetic.mul_inplace(const_cast<void*>(data()), other.data());
        } else {
            RPY_THROW(std::runtime_error, "Type mismatch error.");
        }

        return *this;
    }

    Reference& operator/=(ConstReference other)
    {
        const auto& arithmetic = type()->arithmetic(other.type());
        RPY_CHECK(arithmetic.div_inplace != nullptr);
        if (type() == other.type()) {
            arithmetic.div_inplace(const_cast<void*>(data()), other.data());
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
    return ConstReference(
            is_inline_stored() ? m_storage.bytes : m_storage.pointer,
            p_type
    );
}

inline Value::operator Reference() noexcept
{
    return Reference(
            is_inline_stored() ? m_storage.bytes : m_storage.pointer,
            p_type
    );
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

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_VALUE_H

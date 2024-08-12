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
class Value;

template <typename T>
inline constexpr bool is_reference_like = is_same_v<decay_t<T>, ConstReference>
        || is_same_v<decay_t<T>, Reference>;

template <typename T>
inline constexpr bool value_like
        = is_same_v<decay_t<T>, Value> || is_same_v<decay_t<T>, ConstReference>
        || is_same_v<decay_t<T>, Reference>;

namespace dtl {

template <typename T>
using value_like_return = enable_if_t<value_like<T>, Value>;

struct ValueStorage {

    union Storage
    {
        constexpr Storage() : pointer(nullptr) {}

        alignas(void*) byte bytes[sizeof(void*)];
        void* pointer;
    };

    Storage m_storage;

    RPY_NO_DISCARD static bool is_inline_stored(const Type* type) noexcept
    {
        return type != nullptr && traits::is_arithmetic(*type)
                && size_of(*type) <= sizeof(void*);
    }

    void* data(const Type* type) noexcept
    {
        if (is_inline_stored(type)) { return m_storage.bytes; }
        return m_storage.pointer;
    }

    const void* data(const Type* type) const noexcept
    {
        if (is_inline_stored(type)) { return m_storage.bytes; }
        return m_storage.pointer;
    }
};

}// namespace dtl

class ROUGHPY_DEVICES_EXPORT Value : protected dtl::ValueStorage
{
    friend class ConstReference;
    friend class Reference;

    TypePtr p_type = nullptr;

    RPY_NO_DISCARD bool is_inline_stored() const noexcept
    {
        return ValueStorage::is_inline_stored(p_type.get());
    }

public:
    Value() = default;

    Value(const Value& other);
    Value(Value&& other) noexcept;
    explicit Value(ConstReference other);

    Value(TypePtr type) : p_type(std::move(type))
    {
        if (is_inline_stored()) {
            m_storage.pointer = p_type->allocate_single();
        }
    }

    template <typename T>
    Value(TypePtr type, T&& val) : p_type(std::move(type))
    {
        operator=(std::forward<T>(val));
    }

    template <
            typename T,
            typename = enable_if_t<
                    !is_same_v<decay_t<T>, Value>
                    || !is_same_v<decay_t<T>, ConstReference>
                    || !is_same_v<decay_t<T>, Reference>>>
    explicit Value(T&& other) noexcept : p_type(get_type<T>())
    {
        decay_t<T>* this_ptr;
        if (is_inline_stored()) {
            this_ptr = reinterpret_cast<T*>(m_storage.bytes);
        } else {
            this_ptr = static_cast<T*>(p_type->allocate_single());
            m_storage.pointer = this_ptr;
        }
        construct_inplace(this_ptr, std::forward<T>(other));
    }

    ~Value();

    RPY_NO_DISCARD TypePtr type() const noexcept { return p_type; }

    RPY_NO_DISCARD const void* data() const noexcept
    {
        return ValueStorage::data(&*p_type);
    }

    template <typename T>
    RPY_NO_DISCARD enable_if_t<!is_void_v<T>, const T*> data() const noexcept
    {
        return launder(static_cast<const T*>(ValueStorage::data(&*p_type)));
    }

    RPY_NO_DISCARD void* data() noexcept
    {
        return ValueStorage::data(&*p_type);
    }

    template <typename T>
    RPY_NO_DISCARD enable_if_t<!is_void_v<T>, T*> data() noexcept
    {
        return launder(static_cast<T*>(ValueStorage::data(&*p_type)));
    }

    template <typename T>
    constexpr add_const_t<T>& value() const
    {
        return *data<add_const_t<T>>();
    }

    Value& operator=(const Value& other);
    Value& operator=(Value&& other) noexcept;

    template <typename T>
    enable_if_t<!value_like<T>, Value&> operator=(T&& other);

    template <typename T>
    enable_if_t<is_reference_like<T>, Value&> operator=(const T& other);

    void change_type(const Type* new_type);

    RPY_NO_DISCARD bool fast_is_zero() const noexcept
    {
        return p_type == nullptr || m_storage.pointer == nullptr;
    }
    RPY_NO_DISCARD bool is_zero() const;

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
    enable_if_t<!value_like<T>, Value&> operator+=(const T& other);
    template <typename T>
    enable_if_t<!value_like<T>, Value&> operator-=(const T& other);
    template <typename T>
    enable_if_t<!value_like<T>, Value&> operator*=(const T& other);
    template <typename T>
    enable_if_t<!value_like<T>, Value&> operator/=(const T& other);
};

class ConstReference
{
    const void* p_val;
    TypePtr p_type;

public:
    ConstReference(const void* val, TypePtr type)
        : p_val(val),
          p_type(std::move(type))
    {
        RPY_CHECK(p_type != nullptr);
    }

    RPY_NO_DISCARD TypePtr type() const noexcept { return p_type; }
    RPY_NO_DISCARD const void* data() const noexcept { return p_val; }
    template <typename T>
    RPY_NO_DISCARD enable_if_t<!is_void_v<T>, const T*> data() const noexcept
    {
        return launder(static_cast<const T*>(p_val));
    }

    template <typename T>
    constexpr explicit operator TypedConstReference<T>() const noexcept
    {
        return TypedConstReference<T>(value<T>());
    }

    template <typename T>
    constexpr add_const_t<T>& value() const
    {
        return *data<add_const_t<T>>();
    }
};

class ConstPointer : ConstReference
{
public:
    using ConstReference::ConstReference;

    ConstReference operator*() const noexcept
    {
        return static_cast<ConstReference>(*this);
    }

    const ConstReference* operator->() const noexcept
    {
        return static_cast<const ConstReference*>(this);
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
    Reference(void* val, TypePtr type) : ConstReference(val, std::move(type)) {}

    using ConstReference::data;

    template <typename T>
    enable_if_t<!value_like<T>, Reference&> operator=(T&& other);

    Reference& operator=(const ConstReference& other);
    Reference& operator=(const Value& other);
    Reference& operator=(Value&& other);

    RPY_NO_DISCARD void* data() const noexcept
    {
        return const_cast<void*>(ConstReference::data());
    }

    template <typename T>
    RPY_NO_DISCARD enable_if_t<!is_void_v<T>, T*> data() const noexcept
    {
        return const_cast<T*>(ConstReference::template data<T>());
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
        const auto& arithmetic = type()->arithmetic(*other.type());
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
        const auto& arithmetic = type()->arithmetic(*other.type());
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
        const auto& arithmetic = type()->arithmetic(*other.type());
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
        const auto& arithmetic = type()->arithmetic(*other.type());
        RPY_CHECK(arithmetic.div_inplace != nullptr);
        if (type() == other.type()) {
            arithmetic.div_inplace(data(), other.data());
        } else {
            RPY_THROW(std::runtime_error, "Type mismatch error.");
        }

        return *this;
    }
};

class Pointer : Reference
{
public:
    using Reference::Reference;

    Reference operator*() noexcept { return static_cast<Reference>(*this); }
    ConstReference operator*() const noexcept
    {
        return static_cast<ConstReference>(*this);
    }

    Reference* operator->() noexcept { return static_cast<Reference*>(this); }
    const ConstReference* operator->() const noexcept
    {
        return static_cast<const ConstReference*>(this);
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

inline bool Value::is_zero() const
{
    if (fast_is_zero()) { return true; }
    const auto& comparisons = type()->comparisons(*type());
    if (comparisons.is_zero) { return comparisons.is_zero(data()); }
    if (comparisons.equals) {
        return comparisons.equals(data(), type()->zero().data());
    }
    RPY_THROW(std::runtime_error, "no comparison to zero is available");
}

template <typename T>
enable_if_t<!value_like<T>, Value&> Value::operator=(T&& other)
{
    if (p_type) {
        if (!is_inline_stored() && m_storage.pointer == nullptr) {
            m_storage.pointer = p_type->allocate_single();
        }
        // Convert the value of other to the current type
        const auto& conversion = p_type->conversions(*get_type<T>());
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
enable_if_t<is_reference_like<T>, Value&> Value::operator=(const T& other)
{
    Reference this_val(*this);
    this_val = other;
    return *this;
}

template <typename T>
enable_if_t<!value_like<T>, Reference&> Reference::operator=(T&& other)
{
    RPY_DBG_ASSERT(type() != nullptr);
    const auto tp = get_type<T>();
    const auto& conversion = type()->conversions(*tp);
    if constexpr (is_rvalue_reference_v<T&&>) {
        if (conversion.move_convert) {
            conversion.move_convert(data(), &other);
            return *this;
        }
    }
    conversion.convert(data(), &other);
    return *this;
}

namespace dtl {

struct AdditionCheck {
    bool check(const type_support::TypeArithmetic* impl) const
    {
        RPY_CHECK(impl->add_inplace != nullptr);
        return true;
    }
};

// Check struct for subtraction
struct SubtractionCheck {
    bool check(const type_support::TypeArithmetic* impl) const
    {
        RPY_CHECK(impl->sub_inplace != nullptr);
        return true;
    }
};

// Check struct for multiplication
struct MultiplicationCheck {
    bool check(const type_support::TypeArithmetic* impl) const
    {
        RPY_CHECK(impl->mul_inplace != nullptr);
        return true;
    }
};

// Check struct for division
struct DivisionCheck {
    bool check(const type_support::TypeArithmetic* impl) const
    {
        RPY_CHECK(impl->div_inplace != nullptr);
        return true;
    }
};
}// namespace dtl

inline constexpr dtl::AdditionCheck check_addition;
inline constexpr dtl::SubtractionCheck check_subtraction;
inline constexpr dtl::MultiplicationCheck check_multiplication;
inline constexpr dtl::DivisionCheck check_division;

/**
 * @brief Helper for performing arithmetic between generic values
 *
 * The arithmetic operations on Values/Referencs/ConstReferences do not scale
 * particularly well because each operation incurs the cost of a blocking wait
 * on the Type's Mutex lock in order to get the arithemetic traits. This class
 * is designed to circumvent this cost by performing the locked lookup once, and
 * then using the same instance of the arithemetic trait for all the
 * calculations
 *
 * This is intended to be used when defining generic kernel definitions via a
 * captured parameter or similarly.
 */
class Arithmetic
{
    const type_support::TypeArithmetic* p_impl;

public:
    template <typename... Checks>
    explicit Arithmetic(const Type* main, const Checks&... checks)
        : p_impl(&main->arithmetic(*main))
    {
        RPY_CHECK(... && checks.check(p_impl));
    }

    template <typename... Checks>
    explicit Arithmetic(
            const Type* primary,
            const Type* secondary,
            const Checks&... checks
    )
        : p_impl(&primary->arithmetic(*secondary))
    {
        RPY_CHECK(... && checks.check(p_impl));
    }

    RPY_NO_DISCARD RPY_INLINE_ALWAYS Value
    add(ConstReference left, ConstReference right) const
    {
        Value result(left);
        p_impl->add_inplace(result.data(), right.data());
        return result;
    }

    RPY_NO_DISCARD RPY_INLINE_ALWAYS Value
    sub(ConstReference left, ConstReference right) const
    {
        Value result(left);
        p_impl->sub_inplace(result.data(), right.data());
        return result;
    }

    RPY_NO_DISCARD RPY_INLINE_ALWAYS Value
    mul(ConstReference left, ConstReference right) const
    {
        Value result(left);
        p_impl->mul_inplace(result.data(), right.data());
        return result;
    }

    RPY_NO_DISCARD RPY_INLINE_ALWAYS Value
    div(ConstReference left, ConstReference right) const
    {
        Value result(left);
        p_impl->div_inplace(result.data(), right.data());
        return result;
    }

    RPY_INLINE_ALWAYS void
    add_inplace(Reference left, ConstReference right) const
    {
        p_impl->add_inplace(left.data(), right.data());
    }

    RPY_INLINE_ALWAYS void
    sub_inplace(Reference left, ConstReference right) const
    {
        p_impl->sub_inplace(left.data(), right.data());
    }

    RPY_INLINE_ALWAYS void
    mul_inplace(Reference left, ConstReference right) const
    {
        p_impl->mul_inplace(left.data(), right.data());
    }

    RPY_INLINE_ALWAYS void
    div_inplace(Reference left, ConstReference right) const
    {
        p_impl->div_inplace(left.data(), right.data());
    }
};

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
inline Value& Value::operator+=(const ConstReference other)
{
    const auto& arithmetic = p_type->arithmetic(*other.type());
    RPY_CHECK(arithmetic.add_inplace != nullptr);
    arithmetic.add_inplace(this->data(), other.data());
    return *this;
}

inline Value& Value::operator-=(const ConstReference other)
{
    const auto& arithmetic = p_type->arithmetic(*other.type());
    RPY_CHECK(arithmetic.sub_inplace != nullptr);
    arithmetic.sub_inplace(this->data(), other.data());
    return *this;
}

inline Value& Value::operator*=(const ConstReference other)
{
    const auto& arithmetic = p_type->arithmetic(*other.type());
    RPY_CHECK(arithmetic.mul_inplace != nullptr);
    arithmetic.mul_inplace(this->data(), other.data());
    return *this;
}

inline Value& Value::operator/=(const ConstReference other)
{
    const auto& arithmetic = p_type->arithmetic(*other.type());
    RPY_CHECK(arithmetic.div_inplace != nullptr);
    arithmetic.div_inplace(this->data(), other.data());
    return *this;
}

template <typename T>
enable_if_t<!value_like<T>, Value&> Value::operator+=(const T& other)
{
    if (other != T()) { operator+=(ConstReference(&other, get_type<T>())); }
    return *this;
}
template <typename T>
enable_if_t<!value_like<T>, Value&> Value::operator-=(const T& other)
{
    if (other != T()) { operator-=(ConstReference(&other, get_type<T>())); }
    return *this;
}
template <typename T>
enable_if_t<!value_like<T>, Value&> Value::operator*=(const T& other)
{
    if (other != T()) {
        operator*=(ConstReference(&other, get_type<T>()));
    } else {
        operator=(T());
    }
    return *this;
}
template <typename T>
enable_if_t<!value_like<T>, Value&> Value::operator/=(const T& other)
{
    if (other == T()) { RPY_THROW(std::domain_error, "division by zero"); }
    return operator/=(ConstReference(&other, get_type<T>()));
}

template <typename T>
dtl::value_like_return<T> operator-(const T& arg)
{
    Value result(arg.type(), 0);
    result -= arg;
    return result;
}

template <typename S, typename T>
enable_if_t<value_like<S> && value_like<T>, Value>
operator+(S&& left, const T& right)
{
    Value result(std::forward<S>(left));
    result += right;
    return result;
}

template <typename S, typename T>
enable_if_t<value_like<S> && value_like<T>, Value>
operator-(S&& left, const T& right)
{
    Value result(std::forward<S>(left));
    result -= right;
    return result;
}
template <typename S, typename T>
enable_if_t<value_like<S> && value_like<T>, Value>
operator*(S&& left, const T& right)
{
    Value result(std::forward<S>(left));
    result *= right;
    return result;
}
template <typename S, typename T>
enable_if_t<value_like<S> && value_like<T>, Value>
operator/(S&& left, const T& right)
{
    Value result(std::forward<S>(left));
    result /= right;
    return result;
}

template <typename S, typename T>
enable_if_t<value_like<S> && value_like<T>, bool>
operator==(const S& left, const T& right)
{
    const auto& comparisons = left.type()->comparisons(*right.type());
    RPY_CHECK(comparisons.equals);
    return comparisons.equals(left.data(), right.data());
}

template <typename S, typename T>
enable_if_t<value_like<S> && value_like<T>, bool>
operator!=(const S& left, const T& right)
{
    const auto& comparisons = left.type()->comparisons(*right.type());
    RPY_CHECK(comparisons.equals);
    return !comparisons.equal(left.data(), right.data());
}

template <typename S, typename T>
enable_if_t<value_like<S> && value_like<T>, bool>
operator<(const S& left, const T& right)
{
    const auto& comparisons = left.type()->comparisons(*right.type());
    RPY_CHECK(comparisons.less);
    return comparisons.less(left.data(), right.data());
}

template <typename S, typename T>
enable_if_t<value_like<S> && value_like<T>, bool>
operator<=(const S& left, const T& right)
{
    const auto& comparisons = left.type()->comparisons(*right.type());
    RPY_CHECK(comparisons.less_equal);
    return comparisons.less_equal(left.data(), right.data());
}

template <typename S, typename T>
enable_if_t<value_like<S> && value_like<T>, bool>
operator>(const S& left, const T& right)
{
    const auto& comparisons = left.type()->comparisons(*right.type());
    RPY_CHECK(comparisons.greater);
    return comparisons.greater(left.data(), right.data());
}

template <typename S, typename T>
enable_if_t<value_like<S> && value_like<T>, bool>
operator>=(const S& left, const T& right)
{
    const auto& comparisons = left.type()->comparisons(*right.type());
    RPY_CHECK(comparisons.greater_equal);
    return comparisons.greater_equal(left.data(), right.data());
}

template <typename T>
enable_if_t<value_like<T>, std::ostream&>
operator<<(std::ostream& os, const T& value)
{
    value.type()->display(os, value.data());
    return os;
}

namespace math {

/**
 * @brief Compute the absolute value of a given value.
 *
 * @param value The input value.
 * @return The absolute value of the input value.
 */
template <typename T>
inline dtl::value_like_return<T> abs(const T& value)
{
    TypePtr tp = value.type();
    if (tp == nullptr) { return Value(); }

    const auto* num_traits = tp->num_traits();
    RPY_CHECK(
            num_traits != nullptr && num_traits->abs != nullptr
            && num_traits->real_type != nullptr
    );
    RPY_DBG_ASSERT(num_traits->real_type != nullptr);
    Value result(num_traits->real_type, 0);
    num_traits->abs(result.data(), value.data());
    return result;
}

//! sqrt function
/**
 * @brief Compute the square root of a given value.
 *
 * @param value The input value for which the square root needs to be computed.
 * @return The square root of the input value.
 */
template <typename T>
dtl::value_like_return<T> sqrt(const T& value)
{
    TypePtr tp = value.type();
    if (tp == nullptr) { return Value(); }

    const auto* num_traits = tp->num_traits();

    RPY_CHECK(num_traits != nullptr && num_traits->sqrt != nullptr);

    Value result(tp, 0);
    num_traits->sqrt(result.data(), value.data());

    return result;
}

/**
 * @brief Extracts the real part of a given value.
 *
 * This function calculates the real part of the given value using the
 * num_traits of its type. If the given value is not of a numeric type that has
 * a real part, the function returns a default constructed value.
 *
 * @tparam T The type of the value.
 * @param value The value from which to extract the real part.
 * @return The real part of the value.
 */
template <typename T>
dtl::value_like_return<T> real(const T& value)
{
    TypePtr tp = value.type();
    if (tp == nullptr) { return Value(); }

    const auto* num_traits = tp->num_traits();

    RPY_CHECK(num_traits != nullptr && num_traits->real != nullptr);
    RPY_DBG_ASSERT(num_traits->real_type != nullptr);

    Value result(num_traits->real_type, 0);
    num_traits->real(result.data(), value.data());

    return result;
}

template <typename T>
/**
 * @brief Calculates the imaginary part of a given value.
 *
 * @param value The value for which the imaginary part needs to be calculated.
 * @return A value containing the imaginary part.
 */
dtl::value_like_return<T> imag(const T& value)
{
    TypePtr tp = value.type();
    if (tp == nullptr) { return Value(); }

    const auto* num_traits = tp->num_traits();

    RPY_CHECK(num_traits != nullptr && num_traits->imag != nullptr);
    RPY_DBG_ASSERT(num_traits->imag_type != nullptr);

    Value result(num_traits->imag_type, 0);
    num_traits->imag(result.data(), value.data());

    return result;
}

/**
 * @brief Calculates the complex conjugate of a given value.
 *
 * This method calculates the conjugate of a given value. The conjugate of a
 * complex number is obtained by changing the sign of its imaginary part. For
 * other types, this method returns the value unchanged.
 *
 * @param value The value for which the conjugate needs to be calculated.
 * @return A value containing the conjugate of the given value.
 */
template <typename T>
dtl::value_like_return<T> conj(const T& value)
{
    TypePtr tp = value.type();
    if (tp == nullptr) { return Value(); }

    const auto* num_traits = tp->num_traits();

    RPY_CHECK(num_traits != nullptr && num_traits->conj != nullptr);

    Value result(tp, 0);
    num_traits->conj(result.data(), value.data());

    return result;
}

/**
 * @brief Calculates the power of a given value.
 *
 * This function calculates the power of a given value to the specified power.
 *
 * @param value The value for which the power needs to be calculated.
 * @param power The power to calculate.
 *
 * @return A value containing the result of the power calculation.
 *
 * @pre The type of the value must support the pow operation.
 * @post The returned value will be of the same type as the input value.
 */
template <typename T>
dtl::value_like_return<T> pow(const T& value, unsigned power)
{
    TypePtr tp = value.type();
    if (tp == nullptr) { return Value(); }
    const auto* num_traits = tp->num_traits();

    RPY_CHECK(num_traits != nullptr && num_traits->pow != nullptr);

    Value result(tp, 0);
    if (power == 0) {
        result = 1;
    } else if (power == 1) {
        result = value;
    } else {
        num_traits->pow(result.data(), value.data(), power);
    }

    return result;
}

/**
 * @brief Calculates the exponential value of a given number.
 *
 * This method calculates the exponential value of a given number by using the
 * exp function provided by the num_traits of the value's type. If the
 * num_traits or exp function is not available for the given value's type, the
 * method will return a default Value object.
 *
 * @param value The value for which the exponential value needs to be
 * calculated.
 * @return A Value object containing the exponential value of the given number.
 */
template <typename T>
dtl::value_like_return<T> exp(const T& value)
{
    TypePtr tp = value.type();
    if (tp == nullptr) { return Value(); }

    const auto* num_traits = tp->num_traits();

    RPY_CHECK(num_traits != nullptr && num_traits->exp != nullptr);

    Value result(tp, 0);
    num_traits->exp(result.data(), value.data());

    return result;
}

/**
 * @brief Calculates the logarithm of a given value.
 *
 * This method calculates the logarithm of the given value using the logarithm
 * function defined in the num_traits for the value's type.
 *
 * @param value The value for which the logarithm needs to be calculated.
 * @return A value containing the logarithm of the given value.
 * @remarks If the type of value is not supported or if the value is invalid, an
 * empty Value is returned.
 */
template <typename T>
dtl::value_like_return<T> log(const T& value)
{
    TypePtr tp = value.type();
    if (tp == nullptr) { return Value(); }

    const auto* num_traits = tp->num_traits();

    RPY_CHECK(num_traits != nullptr && num_traits->log != nullptr);

    Value result(tp, 0);
    num_traits->log(result.data(), value.data());

    return result;
}

template <typename T>
dtl::value_like_return<T> reciprocal(T&& val)
{
    Value result(val.type()->one());
    result /= val;
    return result;
}

}// namespace math

template <typename T, typename V>
enable_if_t<value_like<V>, const T&> value_cast(const V& val)
{
    RPY_CHECK(val.type() == get_type<T>());
    return val.template value<T>();
}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_VALUE_H

//
// Created by sam on 14/11/24.
//

#ifndef ROUGHPY_GENERICS_VALUES_H
#define ROUGHPY_GENERICS_VALUES_H

#include <iosfwd>

#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/traits.h"
#include "roughpy/core/types.h"

#include "roughpy/platform/roughpy_platform_export.h"

#include "type.h"
#include "type_ptr.h"

#include "builtin_trait.h"

namespace rpy::generics {
class ConstRef;
class ConstPtr;
class Ref;
class Ptr;
class Value;

namespace dtl {
template<typename T>
inline constexpr bool reference_like_v
        = is_same_v<ConstRef, decay_t<T> > || is_base_of_v<ConstRef,
              decay_t<T> >;

template<typename T>
inline constexpr bool value_like_v = is_same_v<Value, decay_t<T> >
                                     || is_base_of_v<Value, decay_t<T> >
                                     || reference_like_v<T>;
} // namespace dtl

/**
 * @brief Constant reference wrapper for an object.
 *
 * This class provides a way to encapsulate a constant reference to an object.
 * It ensures that the referenced object is not modified, maintaining its
 * const correctness.
 *
 * A ConstRef is only constructible from an existing object with a well-defined
 * type, both of which must remain valid whilst the ConstRef is alive. The
 * ConstRef itself does not take (shared) ownership of the type, as it is the
 * constructor's responsibility to ensure the pointer is valid.
 */
class ROUGHPY_PLATFORM_EXPORT ConstRef {
    const Type *p_type;
    const void *p_data;

protected:
    struct without_null_check {
    };

    // ConstReference should not usually be constructed without
    // valid data, but internally this is a valid state. This
    // constructed is used internally for constructing (derived classes)
    // where the data pointer might be null. (See ConstPointer below.)
    // This causes the construct to skip the validity check.
    ConstRef(const Type *type, const void *data, without_null_check)
        : p_type(type),
          p_data(data) {
    }

    template<typename T = void>
    void set_pointer(const T *ptr) noexcept(is_void_v<T>) {
        if constexpr (!is_void_v<T>) {
            RPY_CHECK(p_type->type_info() == typeid(T));
        }
        p_data = ptr;
    }

public:
    using value_type = Value;
    using const_reference = ConstRef;
    using reference = ConstRef;
    using pointer = ConstPtr;

    ConstRef(const Type *type, const void *p_data)
        : p_type(type),
          p_data(p_data) {
        RPY_CHECK_NE(p_data, nullptr);
    }

    RPY_NO_DISCARD bool is_valid() const noexcept {
        return p_type != nullptr;
    }

    RPY_NO_DISCARD bool fast_is_zero() const noexcept {
        return !is_valid() || p_data == nullptr;
    }

    RPY_NO_DISCARD const Type &type() const noexcept { return *p_type; }

    RPY_NO_DISCARD const Type *type_ptr() const noexcept {
        return p_type;
    }

    template<typename T = void>
    RPY_NO_DISCARD constexpr const T *data() const noexcept {
        if constexpr (is_void_v<T>) {
            return p_data;
        } else {
            RPY_DBG_ASSERT(p_type->type_info() == typeid(T));
            return std::launder(static_cast<const T *>(p_data));
        }
    }
};

class ROUGHPY_PLATFORM_EXPORT ConstPtr : ConstRef {
public:
    using value_type = Value;
    using const_reference = ConstRef;
    using reference = ConstRef;
    using pointer = ConstPtr;

    explicit ConstPtr(const Type *type, const void *data = nullptr)
        : ConstRef(type, data, without_null_check{}) {
    }

    RPY_NO_DISCARD ConstRef operator*() const noexcept {
        RPY_DBG_ASSERT(is_valid());
        return static_cast<ConstRef>(*this);
    }

    RPY_NO_DISCARD const ConstRef *operator->() const noexcept {
        RPY_DBG_ASSERT(is_valid());
        return static_cast<const ConstRef *>(this);
    }
};

/**
 * @brief Mutable reference wrapper for an object.
 *
 * This class extends the ConstRef class to provide mutable access to an
 * encapsulated object. It allows modification of the referenced object and
 * supports in-place arithmetic operations.
 *
 * A Ref object is initialized with a type and data pointer. It provides
 * template methods to retrieve the data pointer as a specific type and ensure
 * const correctness when required.
 */
class ROUGHPY_PLATFORM_EXPORT Ref : public ConstRef {
protected:
    Ref(const Type *type, void *data, without_null_check tag)
        : ConstRef(type, data, tag) {
    }

public:
    Ref(const Type *type, void *data) : ConstRef(type, data) {
    }

    using ConstRef::data;

    template<typename T = void>
    RPY_NO_DISCARD T *data() const noexcept {
        return const_cast<T *>(ConstRef::data<T>());
    }

    // Inplace arithmetic operations
    template<typename T>
    enable_if_t<dtl::value_like_v<T>, Ref &> operator+=(const T &other);

    template<typename T>
    enable_if_t<dtl::value_like_v<T>, Ref &> operator-=(const T &other);

    template<typename T>
    enable_if_t<dtl::value_like_v<T>, Ref &> operator*=(const T &other);

    template<typename T>
    enable_if_t<dtl::value_like_v<T>, Ref &> operator/=(const T &other);
};

class ROUGHPY_PLATFORM_EXPORT Ptr : Ref {
public:
    using value_type = Value;
    using const_reference = ConstRef;
    using reference = Ref;
    using pointer = Ptr;
    using const_pointer = ConstPtr;

    explicit Ptr(const Type *type, void *data = nullptr)
        : Ref(type, data, without_null_check{}) {
    }

    // ReSharper disable once CppNonExplicitConversionOperator
    operator ConstPtr() const noexcept // NOLINT(*-explicit-constructor)
    {
        return ConstPtr{&type(), data()};
    }

    Ref operator*() const noexcept {
        RPY_DBG_ASSERT(is_valid());
        return static_cast<Ref>(*this);
    }

    const Ref *operator->() const noexcept {
        RPY_DBG_ASSERT(is_valid());
        return static_cast<const Ref *>(this);
    }
};

namespace dtl {
class ValueStorage {
    union Storage {
        constexpr Storage() : pointer(nullptr) {
        }

        alignas(void *) byte bytes[sizeof(void *)];
        void *pointer;
    };

    Storage m_storage;
    // NOLINT(*-non-private-member-variables-in-classes)

public:
    constexpr ValueStorage() = default;

    constexpr ValueStorage(ValueStorage &&other) noexcept
        : m_storage(other.m_storage) {
        other.m_storage.pointer = nullptr;
    }

    ValueStorage &operator=(ValueStorage &&other) noexcept {
        if (&other != this) {
            m_storage.pointer = other.m_storage.pointer;
        }
        return *this;
    }

    RPY_NO_DISCARD static bool is_inline_stored(
        const Type *type) noexcept {
        return type != nullptr && concepts::is_arithmetic(*type)
               && size_of(*type) <= sizeof(void *);
    }

    void *data(const Type *type) noexcept {
        if (is_inline_stored(type)) { return m_storage.bytes; }
        return m_storage.pointer;
    }

    const void *data(const Type *type) const noexcept {
        if (is_inline_stored(type)) { return m_storage.bytes; }
        return m_storage.pointer;
    }

    void *reset(void *new_ptr) {
        return std::exchange(m_storage.pointer, new_ptr);
    }
};
} // namespace dtl

/**
 * @brief Represents a generic value in the system.
 *
 * The Value class is designed to encapsulate different types of values used
 * throughout the application. It provides necessary interfaces to handle the
 * encapsulated value in a safe and type-agnostic manner.
 *
 * The exact type of the value encapsulated by this class is determined at the
 * time of instantiation and remains immutable. Operations permitted on the
 * Value object align with the type of the encapsulated value.
 */
class ROUGHPY_PLATFORM_EXPORT Value {
    TypePtr p_type = nullptr;
    dtl::ValueStorage m_storage;

    RPY_NO_DISCARD bool is_inline_stored() const noexcept {
        return dtl::ValueStorage::is_inline_stored(p_type.get());
    }

    friend class ConstRef;
    friend class Ref;

    void allocate_data();

    void
    copy_assign_value(const Type *type, const void *source_data);

    void move_assign_value(const Type *type, void *source_data);

    void ensure_constructed(const Type *backup_type = nullptr);

public:
    using value_type = Value;
    using const_reference = ConstRef;
    using reference = Ref;
    using pointer = Ptr;
    using const_pointer = ConstPtr;

    // standard constructors
    Value();

    Value(const Value &other);

    Value(Value &&other) noexcept;

    // Construct a zero object if this is valid
    explicit Value(TypePtr type, const void *data = nullptr);

    explicit Value(TypePtr type, string_view data);

    // Copy a value from an existing reference
    explicit Value(ConstRef other);

    template<
        typename T,
        typename = enable_if_t<
            !dtl::value_like_v<decay_t<T> >
            && !is_same_v<decay_t<T>, TypePtr>
            && !is_same_v<decay_t<T>, const Type *>> >
    explicit Value(T &&value);

    static Value from_string(TypePtr type, string_view data);

    ~Value();

    // The semantics of copy assignment are different depending on whether the
    // value is initialized or not. If the value is initialized then the copy
    // assignment performs a converting copy from other into this. If the type
    // is not initialized, an ordinary construction is performed.
    Value &operator=(const Value &other);

    // Move construction also has different semantics
    // ReSharper disable once CppSpecialFunctionWithoutNoexceptSpecification
    Value &operator=(Value &&other); // NOLINT(*-noexcept-move-constructor)

    Value &operator=(ConstRef other);

    operator ConstRef() const noexcept {
        return {p_type.get(), data()};
    }

    operator Ref() noexcept {
        return {p_type.get(), data()};
    }

    RPY_NO_DISCARD bool is_valid() const noexcept {
        return static_cast<bool>(p_type);
    }

    RPY_NO_DISCARD const Type *type_ptr() const noexcept {
        return p_type.get();
    }

    RPY_NO_DISCARD const Type &type() const noexcept {
        RPY_DBG_ASSERT(is_valid());
        return *p_type;
    }

    template<typename T = void>
    RPY_NO_DISCARD const T *data() const noexcept(is_void_v<T>) {
        if (!p_type) { return nullptr; }

        if constexpr (is_void_v<T>) {
            return m_storage.data(p_type.get());
        } else {
            RPY_CHECK_EQ(p_type->type_info(), typeid(T));
            return std::launder(
                static_cast<const T *>(m_storage.data(p_type.get()))
            );
        }
    }

    template<typename T = void>
    RPY_NO_DISCARD T *data() noexcept(is_void_v<T>) {
        if (!p_type) { return nullptr; }

        if constexpr (is_void_v<T>) {
            return m_storage.data(p_type.get());
        } else {
            RPY_CHECK_EQ(p_type->type_info(), typeid(T));
            return std::launder(
                static_cast<T *>(m_storage.data(p_type.get())));
        }
    }

    // Inplace arithmetic operations
    template<typename T>
    enable_if_t<dtl::value_like_v<T>, Value &> operator+=(const T &other);

    template<typename T>
    enable_if_t<dtl::value_like_v<T>, Value &> operator-=(const T &other);

    template<typename T>
    enable_if_t<dtl::value_like_v<T>, Value &> operator*=(const T &other);

    template<typename T>
    enable_if_t<dtl::value_like_v<T>, Value &> operator/=(const T &other);
};

namespace dtl {
void ROUGHPY_PLATFORM_EXPORT backup_display(std::ostream &os);
}

template<typename T, typename>
Value::Value(T &&value) : p_type(Type::of<T>()) {
    move_assign_value(p_type.get(), &value);
}

template<typename T>
enable_if_t<dtl::value_like_v<T>, std::ostream &>
operator<<(std::ostream &os, const T &value) {
    if (RPY_LIKELY(value.is_valid())) {
        value.type().display(os, value.data());
    } else {
        dtl::backup_display(os);
    }
    return os;
}

template<typename T>
enable_if_t<dtl::value_like_v<T>, hash_t> hash_value(const T &value) {
    return value.type().hash_of(value.data());
}

namespace dtl {
ROUGHPY_PLATFORM_EXPORT
bool values_compare(
    ComparisonType comp,
    const Type *ltype,
    const void *lvalue,
    const Type *rtype,
    const void *rvalue
);
} // namespace dtl

template<typename T, typename U>
enable_if_t<dtl::value_like_v<T> && dtl::value_like_v<U>, bool>
operator==(const T &lhs, const U &rhs) {
    return dtl::values_compare(
        ComparisonType::Equal,
        lhs.type_ptr(),
        lhs.data(),
        rhs.type_ptr(),
        rhs.data()
    );
}

template<typename T, typename U>
enable_if_t<dtl::value_like_v<T> && dtl::value_like_v<U>, bool>
operator!=(const T &lhs, const U &rhs) {
    return !operator==(lhs, rhs);
}

template<typename T, typename U>
enable_if_t<dtl::value_like_v<T> && dtl::value_like_v<U>, bool>
operator<(const T &lhs, const U &rhs) {
    return dtl::values_compare(
        ComparisonType::Less,
        lhs.type_ptr(),
        lhs.data(),
        rhs.type_ptr(),
        rhs.data()
    );
}

template<typename T, typename U>
enable_if_t<dtl::value_like_v<T> && dtl::value_like_v<U>, bool>
operator<=(const T &lhs, const U &rhs) {
    return dtl::values_compare(
        ComparisonType::LessEqual,
        lhs.type_ptr(),
        lhs.data(),
        rhs.type_ptr(),
        rhs.data()
    );
}

template<typename T, typename U>
enable_if_t<dtl::value_like_v<T> && dtl::value_like_v<U>, bool>
operator>(const T &lhs, const U &rhs) {
    return dtl::values_compare(
        ComparisonType::Greater,
        lhs.type_ptr(),
        lhs.data(),
        rhs.type_ptr(),
        rhs.data()
    );
}

template<typename T, typename U>
enable_if_t<dtl::value_like_v<T> && dtl::value_like_v<U>, bool>
operator>=(const T &lhs, const U &rhs) {
    return dtl::values_compare(
        ComparisonType::GreaterEqual,
        lhs.type_ptr(),
        lhs.data(),
        rhs.type_ptr(),
        rhs.data()
    );
}

namespace dtl {
/**
 * @brief Performs arithmetic operations on the given values.
 *
 * This function performs the specified arithmetic operation on two values
 * of potentially different types. The specific operation to be performed
 * is determined by the `operation` parameter.
 *
 * @param operation The arithmetic operation to be performed. It can be one of
 *                  the following: Add, Sub, Mul, Div (defined in the
 * ArithmeticTrait::Operation enum).
 * @param ltype The type of the left-hand side value.
 * @param lvalue A pointer to the left-hand side value.
 * @param rtype The type of the right-hand side value.
 * @param rvalue A pointer to the right-hand side value.
 */
ROUGHPY_PLATFORM_EXPORT void value_inplace_arithmetic(
    ArithmeticOperation operation,
    const Type *ltype,
    void *lvalue,
    const Type *rtype,
    const void *rvalue
);

RPY_NO_DISCARD Value ROUGHPY_PLATFORM_EXPORT value_arithmetic(
    ArithmeticOperation operation,
    const Type *ltype,
    const void *lvalue,
    const Type *rtype,
    const void *rvalue
);
} // namespace dtl


template<typename T>
enable_if_t<dtl::value_like_v<T>, Ref &> Ref::operator+=(const T &other) {
    RPY_CHECK(is_valid() && other.is_valid());

    dtl::value_inplace_arithmetic(
        ArithmeticOperation::Add,
        type_ptr(),
        data(),
        other.type_ptr(),
        other.data()
    );

    return *this;
}

template<typename T>
enable_if_t<dtl::value_like_v<T>, Ref &> Ref::operator-=(const T &other) {
    RPY_CHECK(is_valid() && other.is_valid());

    dtl::value_inplace_arithmetic(
        ArithmeticOperation::Sub,
        type_ptr(),
        data(),
        other.type_ptr(),
        other.data()
    );

    return *this;
}

template<typename T>
enable_if_t<dtl::value_like_v<T>, Ref &> Ref::operator*=(const T &other) {
    RPY_CHECK(is_valid() && other.is_valid());

    dtl::value_inplace_arithmetic(
        ArithmeticOperation::Mul,
        type_ptr(),
        data(),
        other.type_ptr(),
        other.data()
    );
    return *this;
}

template<typename T>
enable_if_t<dtl::value_like_v<T>, Ref &> Ref::operator/=(const T &other) {
    RPY_CHECK(is_valid() && other.is_valid());

    dtl::value_inplace_arithmetic(
        ArithmeticOperation::Div,
        type_ptr(),
        data(),
        other.type_ptr(),
        other.data()
    );

    return *this;
}

template<typename T>
enable_if_t<dtl::value_like_v<T>, Value &> Value::operator
+=(const T &other) {
    RPY_CHECK(other.is_valid());
    ensure_constructed(&other.type());

    dtl::value_inplace_arithmetic(
        ArithmeticOperation::Add,
        type_ptr(),
        data(),
        other.type_ptr(),
        other.data()
    );

    return *this;
}

template<typename T>
enable_if_t<dtl::value_like_v<T>, Value &> Value::operator
-=(const T &other) {
    RPY_CHECK(other.is_valid());
    ensure_constructed(&other.type());

    dtl::value_inplace_arithmetic(
        ArithmeticOperation::Sub,
        type_ptr(),
        data(),
        other.type_ptr(),
        other.data()
    );
    return *this;
}

template<typename T>
enable_if_t<dtl::value_like_v<T>, Value &> Value::operator
*=(const T &other) {
    RPY_CHECK(other.is_valid());
    ensure_constructed(&other.type());

    dtl::value_inplace_arithmetic(
        ArithmeticOperation::Mul,
        type_ptr(),
        data(),
        other.type_ptr(),
        other.data()
    );
    return *this;
}

template<typename T>
enable_if_t<dtl::value_like_v<T>, Value &> Value::operator
/=(const T &other) {
    RPY_CHECK(other.is_valid());
    ensure_constructed(&other.type());

    dtl::value_inplace_arithmetic(
        ArithmeticOperation::Div,
        type_ptr(),
        data(),
        other.type_ptr(),
        other.data()
    );
    return *this;
}

template<typename T>
enable_if_t<dtl::value_like_v<T>, Value> operator-(const T &value) {
    Value result(value.type_ptr());
    if (!value.fast_is_zero()) {
        dtl::value_inplace_arithmetic(
            ArithmeticOperation::Sub,
            result.type_ptr(),
            result.data(),
            value.type_ptr(),
            value.data()
        );
    }
    return result;
}

template<typename T, typename U>
enable_if_t<dtl::value_like_v<T> && dtl::value_like_v<U>, Value>
operator+(const T &lhs, const U &rhs) {
    return dtl::value_arithmetic(
        ArithmeticOperation::Add,
        lhs.type_ptr(),
        lhs.data(),
        rhs.type_ptr(),
        rhs.data()
    );
}

template<typename T, typename U>
enable_if_t<dtl::value_like_v<T> && dtl::value_like_v<U>, Value>
operator-(const T &lhs, const U &rhs) {
    return dtl::value_arithmetic(
        ArithmeticOperation::Sub,
        lhs.type_ptr(),
        lhs.data(),
        rhs.type_ptr(),
        rhs.data()
    );
}

template<typename T, typename U>
enable_if_t<dtl::value_like_v<T> && dtl::value_like_v<U>, Value>
operator*(const T &lhs, const U &rhs) {
    return dtl::value_arithmetic(
        ArithmeticOperation::Mul,
        lhs.type_ptr(),
        lhs.data(),
        rhs.type_ptr(),
        rhs.data()
    );
}

template<typename T, typename U>
enable_if_t<dtl::value_like_v<T> && dtl::value_like_v<U>, Value>
operator/(const T &lhs, const U &rhs) {
    return dtl::value_arithmetic(
        ArithmeticOperation::Div,
        lhs.type_ptr(),
        lhs.data(),
        rhs.type_ptr(),
        rhs.data()
    );
}

namespace dtl {
RPY_NO_DISCARD ROUGHPY_PLATFORM_EXPORT Value
math_fn(NumberFunction func, const Type *type, const void *data);

ROUGHPY_PLATFORM_EXPORT
void from_rational(
    const Type *type,
    void *dst,
    int64_t numerator,
    int64_t denominator
);
} // namespace dtl
namespace math {
template<typename T>
enable_if_t<dtl::value_like_v<T>, Value> abs(const T &value) {
    return dtl::math_fn(NumberFunction::Abs, value.type_ptr(),
                        value.data());
}

template<typename T>
enable_if_t<dtl::value_like_v<T>, Value> sqrt(const T &value) {
    return dtl::math_fn(NumberFunction::Sqrt, value.type_ptr(),
                        value.data());
}

template<typename T>
enable_if_t<dtl::value_like_v<T>, Value> pow(
    const T &value, exponent_t exp) {
    auto payload = std::make_pair(value.data(), exp);
    return dtl::math_fn(
        NumberFunction::Pow,
        value.type_ptr(),
        &payload
    );
}

template<typename T>
enable_if_t<dtl::value_like_v<T>, Value> exp(const T &value) {
    return dtl::math_fn(NumberFunction::Exp, value.type_ptr(),
                        value.data());
}

template<typename T>
enable_if_t<dtl::value_like_v<T>, Value> log(const T &value) {
    return dtl::math_fn(NumberFunction::Log, value.type_ptr(),
                        value.data());
}

template<typename T>
enable_if_t<dtl::value_like_v<T>, Value> real(const T &value) {
    return dtl::math_fn(NumberFunction::Real, value.type_ptr(),
                        value.data());
}

template<typename T>
enable_if_t<dtl::value_like_v<T>, Value> imaginary(const T &value) {
    return dtl::math_fn(
        NumberFunction::Imaginary,
        value.type_ptr(),
        value.data()
    );
}

inline Value from_rational(TypePtr type, int64_t numerator,
                           int64_t denominator) {
    RPY_CHECK(type);
    Value result(std::move(type));
    dtl::from_rational(
        result.type_ptr(),
        result.data(),
        numerator,
        denominator
    );
    return result;
}

inline void from_rational(Ref value, int64_t numerator,
                          int64_t denominator) {
    RPY_CHECK(value.is_valid());
    dtl::from_rational(value.type_ptr(), value.data(), numerator,
                       denominator);
}
} // namespace math
} // namespace rpy::generics

#endif// ROUGHPY_GENERICS_VALUES_H

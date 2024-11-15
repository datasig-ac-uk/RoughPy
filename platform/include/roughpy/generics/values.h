//
// Created by sam on 14/11/24.
//

#ifndef ROUGHPY_GENERICS_VALUES_H
#define ROUGHPY_GENERICS_VALUES_H

#include <iosfwd>

#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/types.h"
#include "roughpy/core/traits.h"

#include "roughpy/platform/roughpy_platform_export.h"

#include "type_ptr.h"
#include "type.h"

#include "arithmetic_trait.h"
#include "builtin_trait.h"
#include "hash_trait.h"


namespace rpy::generics {

class ConstRef;
class ConstPtr;
class Ref;
class Ptr;
class Value;


namespace dtl {

template <typename T>
inline constexpr bool reference_like_v = is_same_v<ConstRef, decay_t<T>>
    || is_base_of_v<ConstRef, decay_t<T>>;

template <typename T>
inline constexpr bool value_like_v =
    is_same_v<Value, decay_t<T>>
    || is_base_of_v<Value, decay_t<T>>
    || reference_like_v<T>;


}



class ROUGHPY_PLATFORM_EXPORT ConstRef {
    TypePtr p_type;
    const void* p_data;

protected:

    struct without_null_check {};

    // ConstReference should not usually be constructed without
    // valid data, but internally this is a valid state. This
    // constructed is used internally for constructing (derived classes)
    // where the data pointer might be null. (See ConstPointer below.)
    // This causes the construct to skip the validity check.
    ConstRef(TypePtr type, const void* data, without_null_check)
        : p_type(std::move(type)), p_data(data) {}

    template <typename T=void>
    void set_pointer(const T* ptr) noexcept(is_void_v<T>)
    {
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


    ConstRef(TypePtr type, const void* p_data)
        : p_type(std::move(type)), p_data(p_data)
    {
        RPY_CHECK_NE(p_data, nullptr);
    }

    static ConstRef zero(TypePtr type)
    {
        RPY_CHECK_NE(type->get_builtin_trait(BuiltinTraitID::Number), nullptr);
        return {std::move(type), nullptr, without_null_check{}};
    }

    RPY_NO_DISCARD
    bool is_valid() const noexcept
    {
        return static_cast<bool>(p_type);
    }

    RPY_NO_DISCARD
    bool fast_is_zero() const noexcept
    {
        return !is_valid() || p_data == nullptr;
    }

    RPY_NO_DISCARD
    const Type& type() const noexcept
    {
        return *p_type;
    }

    template <typename T = void>
    RPY_NO_DISCARD
    constexpr const T* data() const noexcept
    {
        if constexpr (is_void_v<T>) {
            return p_data;
        } else {
            RPY_DBG_ASSERT(p_type->type_info() == typeid(T));
            return std::launder(static_cast<const T*>(p_data));
        }
    }

};

class ROUGHPY_PLATFORM_EXPORT ConstPtr : ConstRef
{

public:

    using value_type = Value;
    using const_reference = ConstRef;
    using reference = ConstRef;
    using pointer = ConstPtr;

    explicit ConstPtr(TypePtr type, const void* data=nullptr)
        : ConstRef(std::move(type), data, without_null_check{})
    {
    }

    RPY_NO_DISCARD
    ConstRef operator*() const noexcept
    {
        RPY_DBG_ASSERT(is_valid());
        return static_cast<ConstRef>(*this);
    }

    RPY_NO_DISCARD
    const ConstRef* operator->() const noexcept
    {
        RPY_DBG_ASSERT(is_valid());
        return static_cast<const ConstRef*>(this);
    }



};


class ROUGHPY_PLATFORM_EXPORT Ref : public ConstRef
{


protected:

    Ref(TypePtr type, void* data, without_null_check tag)
        : ConstRef(std::move(type), data, tag)
    {}

public:

    Ref(TypePtr type, void* data)
        : ConstRef(std::move(type), data)
    {}

    using ConstRef::data;

    template <typename T = void>
    RPY_NO_DISCARD
    T* data() const noexcept
    {
        return const_cast<T*>(ConstRef::data<T>());
    }


    // Inplace arithmetic operations
    template <typename T>
    enable_if_t<dtl::value_like_v<T>, Ref&>
    operator+=(const T& other);

    template <typename T>
    enable_if_t<dtl::value_like_v<T>, Ref&>
    operator-=(const T& other);

    template <typename T>
    enable_if_t<dtl::value_like_v<T>, Ref&>
    operator*=(const T& other);

    template <typename T>
    enable_if_t<dtl::value_like_v<T>, Ref&>
    operator/=(const T& other);
};


class ROUGHPY_PLATFORM_EXPORT Ptr : Ref
{
public:
    using value_type = Value;
    using const_reference = ConstRef;
    using reference = Ref;
    using pointer = Ptr;
    using const_pointer = ConstPtr;

    explicit Ptr(TypePtr type, void* data = nullptr)
        : Ref(std::move(type), data, without_null_check{})
    {}

    // ReSharper disable once CppNonExplicitConversionOperator
    operator ConstPtr() const noexcept // NOLINT(*-explicit-constructor)
    {
        return ConstPtr {&type(), data()};
    }

    Ref operator*() const noexcept
    {
        RPY_DBG_ASSERT(is_valid());
        return static_cast<Ref>(*this);
    }

    const Ref* operator->() const noexcept
    {
        RPY_DBG_ASSERT(is_valid());
        return static_cast<const Ref*>(this);
    }

};



namespace dtl {

class ValueStorage {

    union Storage
    {
        constexpr Storage() : pointer(nullptr) {}

        alignas(void*) byte bytes[sizeof(void*)];
        void* pointer;
    };

    Storage m_storage;// NOLINT(*-non-private-member-variables-in-classes)

public:

    constexpr ValueStorage() = default;

    constexpr ValueStorage(ValueStorage&& other) noexcept
        : m_storage(other.m_storage)
    {
        other.m_storage.pointer = nullptr;
    }

    ValueStorage& operator=(ValueStorage&& other) noexcept
    {
        if (&other != this) {
            m_storage.pointer = other.m_storage.pointer;
        }
        return *this;
    }

    RPY_NO_DISCARD static bool is_inline_stored(const Type* type) noexcept
    {
        return type != nullptr && concepts::is_arithmetic(*type)
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

    void* reset(void* new_ptr)
    {
        return std::exchange(m_storage.pointer, new_ptr);
    }

};

}


class ROUGHPY_PLATFORM_EXPORT Value
{
    TypePtr p_type = nullptr;
    dtl::ValueStorage m_storage;

    RPY_NO_DISCARD bool is_inline_stored() const noexcept
    {
        return dtl::ValueStorage::is_inline_stored(p_type.get());
    }

    friend class ConstRef;
    friend class Ref;

    void allocate_data();

    void assign_value(const Type* type, const void* source_data, bool move=false);

    void ensure_constructed(const Type* backup_type=nullptr);


public:
    // standard constructors
    Value();
    Value(const Value& other);
    Value(Value&& other) noexcept;

    // Construct a zero object if this is valid
    explicit Value(TypePtr type);

    // Copy a value from an existing reference
    explicit Value(ConstRef other);

    template <typename T, typename=enable_if_t<!dtl::value_like_v<T>>>
    explicit Value(T&& value);

    ~Value();

    // The semantics of copy assignment are different depending on whether the
    // value is initialized or not. If the value is initialized then the copy
    // assignment performs a converting copy from other into this. If the type
    // is not initialized, an ordinary construction is performed.
    Value& operator=(const Value& other);

    // Move construction also has different semantics
    // ReSharper disable once CppSpecialFunctionWithoutNoexceptSpecification
    Value& operator=(Value&& other);// NOLINT(*-noexcept-move-constructor)

    Value& operator=(ConstRef other);

    RPY_NO_DISCARD bool is_valid() const noexcept
    {
        return static_cast<bool>(p_type);
    }

    RPY_NO_DISCARD
    const Type& type() const noexcept {
        RPY_DBG_ASSERT(is_valid());
        return *p_type;
    }


    template <typename T = void>
    RPY_NO_DISCARD const T* data() const noexcept(is_void_v<T>)
    {
        if (!p_type) { return nullptr; }

        if constexpr (is_void_v<T>) {
            return m_storage.data(p_type.get());
        } else {
            RPY_CHECK_EQ(p_type->type_info(), typeid(T));
            return std::launder(
                    static_cast<const T*>(m_storage.data(p_type.get()))
            );
        }
    }

    template <typename T = void>
    RPY_NO_DISCARD T* data() noexcept(is_void_v<T>)
    {
        if (!p_type) { return nullptr; }

        if constexpr (is_void_v<T>) {
            return m_storage.data(p_type.get());
        } else {
            RPY_CHECK_EQ(p_type->type_info(), typeid(T));
            return std::launder(static_cast<T*>(m_storage.data(p_type.get())));
        }
    }


    // Inplace arithmetic operations
    template <typename T>
    enable_if_t<dtl::value_like_v<T>, Value&>
    operator+=(const T& other);

    template <typename T>
    enable_if_t<dtl::value_like_v<T>, Value&>
    operator-=(const T& other);

    template <typename T>
    enable_if_t<dtl::value_like_v<T>, Value&>
    operator*=(const T& other);

    template <typename T>
    enable_if_t<dtl::value_like_v<T>, Value&>
    operator/=(const T& other);
};


namespace dtl {

void backup_display(std::ostream& os);

}

template <typename T>
enable_if_t<dtl::value_like_v<T>, std::ostream&>
operator<<(std::ostream& os, const T& value)
{
    if (RPY_LIKELY(value.is_valid())) {
        value.type().display(os, value.data());
    } else {
        dtl::backup_display(os);
    }
    return os;
}


template <typename T>
enable_if_t<dtl::value_like_v<T>, hash_t> hash_value(const Value& value)
{
    const auto* trait = trait_cast<HashTrait>(
            value.type().get_builtin_trait(BuiltinTraitID::Hash)
    );
    RPY_CHECK_NE(trait, nullptr);
    return trait->unsafe_hash(value.data());
}


template <typename T, typename>
Value::Value(T&& value)
    : p_type(get_type<T>())
{
    assign_value(p_type.get(), &value);
}

template <typename T>
enable_if_t<dtl::value_like_v<T>, Ref&> Ref::operator+=(const T& other)
{
    RPY_CHECK(other.is_valid());
    ensure_constructed(&other.type());

    const auto* trait = trait_cast<ArithmeticTrait>(type().get_builtin_trait(BuiltinTraitID::Arithmetic));
    RPY_CHECK_NE(trait, nullptr);
    RPY_CHECK(trait->has_operation(ArithmeticTrait::Operation::Add));

    trait->unsafe_add_inplace(data(), other.data());

    return *this;
}

template <typename T>
enable_if_t<dtl::value_like_v<T>, Ref&> Ref::operator-=(const T& other)
{
    RPY_CHECK(other.is_valid());
    ensure_constructed(&other.type());

    const auto* trait = trait_cast<ArithmeticTrait>(type().get_builtin_trait(BuiltinTraitID::Arithmetic));
    RPY_CHECK_NE(trait, nullptr);
    RPY_CHECK(trait->has_operation(ArithmeticTrait::Operation::Sub));

    trait->unsafe_sub_inplace(data(), other.data());

    return *this;
}
template <typename T>
enable_if_t<dtl::value_like_v<T>, Ref&> Ref::operator*=(const T& other)
{
    RPY_CHECK(other.is_valid());
    ensure_constructed(&other.type());

    const auto* trait = trait_cast<ArithmeticTrait>(type().get_builtin_trait(BuiltinTraitID::Arithmetic));
    RPY_CHECK_NE(trait, nullptr);
    RPY_CHECK(trait->has_operation(ArithmeticTrait::Operation::Mul));

    trait->unsafe_mul_inplace(data(), other.data());
    return *this;
}
template <typename T>
enable_if_t<dtl::value_like_v<T>, Ref&> Ref::operator/=(const T& other)
{
    RPY_CHECK(other.is_valid());
    ensure_constructed(&other.type());

    const auto* trait = trait_cast<ArithmeticTrait>(type().get_builtin_trait(BuiltinTraitID::Arithmetic));
    RPY_CHECK_NE(trait, nullptr);
    RPY_CHECK(trait->has_operation(ArithmeticTrait::Operation::Div));

    trait->unsafe_sub_inplace(data(), other.data());
    return *this;
}

template <typename T>
enable_if_t<dtl::value_like_v<T>, Value&> Value::operator+=(const T& other)
{
    RPY_CHECK(other.is_valid());
    ensure_constructed(&other.type());

    const auto* trait = trait_cast<ArithmeticTrait>(p_type->get_builtin_trait(BuiltinTraitID::Arithmetic));
    RPY_CHECK_NE(trait, nullptr);
    RPY_CHECK(trait->has_operation(ArithmeticTrait::Operation::Add));

    trait->unsafe_add_inplace(data(), other.data());

    return *this;
}

template <typename T>
enable_if_t<dtl::value_like_v<T>, Value&> Value::operator-=(const T& other)
{
    RPY_CHECK(other.is_valid());
    ensure_constructed(&other.type());

    const auto* trait = trait_cast<ArithmeticTrait>(p_type->get_builtin_trait(BuiltinTraitID::Arithmetic));
    RPY_CHECK_NE(trait, nullptr);
    RPY_CHECK(trait->has_operation(ArithmeticTrait::Operation::Sub));

    trait->unsafe_sub_inplace(data(), other.data());

    return *this;
}
template <typename T>
enable_if_t<dtl::value_like_v<T>, Value&> Value::operator*=(const T& other)
{
    RPY_CHECK(other.is_valid());
    ensure_constructed(&other.type());

    const auto* trait = trait_cast<ArithmeticTrait>(p_type->get_builtin_trait(BuiltinTraitID::Arithmetic));
    RPY_CHECK_NE(trait, nullptr);
    RPY_CHECK(trait->has_operation(ArithmeticTrait::Operation::Mul));

    trait->unsafe_mul_inplace(data(), other.data());
    return *this;
}
template <typename T>
enable_if_t<dtl::value_like_v<T>, Value&> Value::operator/=(const T& other)
{
    RPY_CHECK(other.is_valid());
    ensure_constructed(&other.type());

    const auto* trait = trait_cast<ArithmeticTrait>(p_type->get_builtin_trait(BuiltinTraitID::Arithmetic));
    RPY_CHECK_NE(trait, nullptr);
    RPY_CHECK(trait->has_operation(ArithmeticTrait::Operation::Div));

    trait->unsafe_sub_inplace(data(), other.data());
    return *this;
}

}


#endif //ROUGHPY_GENERICS_VALUES_H

//
// Created by sam on 14/11/24.
//

#ifndef ROUGHPY_GENERICS_VALUES_H
#define ROUGHPY_GENERICS_VALUES_H


#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/types.h"
#include "roughpy/core/traits.h"

#include "roughpy/platform/roughpy_platform_export.h"

#include "type_ptr.h"
#include "type.h"

namespace rpy::generics {

class ConstRef;
class ConstPtr;
class Ref;
class Ptr;
class Value;


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

    RPY_NO_DISCARD static bool is_inline_stored(const Type* type) noexcept
    {
        return type != nullptr && type_props::is_arithmetic(*type)
                && type_props::size_of(*type) <= sizeof(void*);
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

    constexpr void* reset(void* new_ptr)
    {
        return std::exchange(m_storage.pointer, new_ptr);
    }

};

}


class ROUGHPY_PLATFORM_EXPORT Value
{
    dtl::ValueStorage m_storage;
    TypePtr p_type = nullptr;

    RPY_NO_DISCARD bool is_inline_stored() const noexcept
    {
        return dtl::ValueStorage::is_inline_stored(p_type.get());
    }

    friend class ConstRef;
    friend class Ref;

    void allocate_data();

    void assign_value(const Type* type, const void* source_data);

public:

    // standard constructors
    Value();
    Value(const Value& other);
    Value(Value&& other) noexcept;

    // Construct a zero object if this is valid
    explicit Value(TypePtr type);

    // Copy a value from an existing reference
    explicit Value(ConstRef other);

    ~Value();

    // The semantics of copy assignment are different depending on whether the
    // value is initialized or not. If the value is initialized then the copy
    // assignment performs a converting copy from other into this. If the type
    // is not initialized, an ordinary construction is performed.
    Value& operator=(const Value& other);

    // Move construction also has different semantics
    // ReSharper disable once CppSpecialFunctionWithoutNoexceptSpecification
    Value& operator=(Value&& other); // NOLINT(*-noexcept-move-constructor)

    Value& operator=(ConstRef other);

    template <typename T = void>
    RPY_NO_DISCARD const T* data() const noexcept(is_void_v<T>)
    {
        if (!p_type) {
            return nullptr;
        }

        if constexpr (is_void_v<T>) {
            return m_storage.data(p_type.get());
        } else {
            RPY_CHECK_EQ(p_type->type_info(), typeid(T));
            return std::launder(
                static_cast<const T*>(m_storage.data(p_type.get())));
        }
    }

    template <typename T = void>
    RPY_NO_DISCARD T* data() noexcept(is_void_v<T>)
    {
        if (!p_type) {
            return nullptr;
        }

        if constexpr (is_void_v<T>) {
            return m_storage.data(p_type.get());
        } else {
            RPY_CHECK_EQ(p_type->type_info(), typeid(T));
            return std::launder(
                static_cast<T*>(m_storage.data(p_type.get())));
        }
    }


    // Inplace arithmetic operations
    Value& operator+=(const Value& other)
    {

        return *this;
    }

    Value& operator-=(const Value& other)
    {

        return *this;
    }

    Value& operator*=(const Value& other)
    {

        return *this;
    }

    Value& operator/=(const Value& other)
    {

        return *this;
    }




};



}


#endif //ROUGHPY_GENERICS_VALUES_H

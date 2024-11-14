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
#include "const_reference.h"
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
    constexpr ConstRef(TypePtr type, const void* data, without_null_check)
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
        RPY_CHECK_NE(type->get_builtin_trait((BuiltinTraitID::Number)), nullptr);
        return {std::move(type), nullptr, without_null_check{}};
    }

    RPY_NO_DISCARD
    constexpr bool is_valid() const noexcept
    {
        return static_cast<bool>(p_type) && p_data != nullptr;
    }

    RPY_NO_DISCARD
    constexpr bool fast_is_zero() const noexcept
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
    constexpr ConstRef operator*() const noexcept
    {
        RPY_DBG_ASSERT(is_valid());
        return static_cast<ConstRef>(*this);
    }

    RPY_NO_DISCARD
    constexpr const ConstRef* operator->() const noexcept
    {
        RPY_DBG_ASSERT(is_valid());
        return static_cast<const ConstRef*>(this);
    }



};


class ROUGHPY_PLATFORM_EXPORT Ref : public ConstRef
{


protected:

    constexpr Ref(TypePtr type, void* data, without_null_check tag)
        : ConstRef(std::move(type), data, tag)
    {}

public:

    Ref(TypePtr type, void* data)
        : ConstRef(std::move(type), data)
    {}

    using ConstRef::data;

    template <typename T = void>
    RPY_NO_DISCARD
    constexpr T* data() const noexcept
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

    explicit Ptr(TypePtr type, const void* data = nullptr)
        : Ref(std::move(type), data, without_null_check{})
    {}

    // ReSharper disable once CppNonExplicitConversionOperator
    operator ConstPtr() const noexcept // NOLINT(*-explicit-constructor)
    {
        return ConstPtr {&type(), data()};
    }

    constexpr Ref operator*() const noexcept
    {
        RPY_DBG_ASSERT(is_valid());
        return static_cast<Ref>(*this);
    }

    constexpr const Ref* operator->() const noexcept
    {
        RPY_DBG_ASSERT(is_valid());
        return static_cast<const Ref*>(this);
    }


};




}


#endif //ROUGHPY_GENERICS_VALUES_H

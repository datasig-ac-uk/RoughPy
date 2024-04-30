//
// Created by sam on 3/29/24.
//

#ifndef ROUGHPY_DEVICES_VALUE_H
#define ROUGHPY_DEVICES_VALUE_H

#include <roughpy/core/traits.h>
#include <utility>

namespace rpy {
namespace devices {

class ConstReference;
template <typename T>
class TypedConstReference;
class Reference;
template <typename T>
class TypedReference;

class ConstReference
{
    const void* p_val;

public:
    constexpr explicit ConstReference(const void* val) : p_val(val) {}

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

class Reference
{
    void* p_val;

public:
    constexpr explicit Reference(void* val) : p_val(val) {}

    constexpr operator ConstReference() const noexcept
    {
        return ConstReference(p_val);
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
        return *static_cast<T*>(p_val);
    }
};

template <typename T>
class TypedReference : public Reference
{
public:
    constexpr TypedReference(T& t)
        : Reference(const_cast<remove_cv_t<T>*>(std::addressof(t)))
    {}

    using Reference::value;

    constexpr operator TypedConstReference<T>() const noexcept
    {
        return TypedConstReference(value<T>());
    }

    constexpr operator add_const_t<T>&() const
    {
        return *value<add_const_t<T>>();
    }

    template <typename U = remove_cv_ref_t<T>>
    constexpr operator enable_if_t<is_convertible_v<T&, U&>, U&>()
    {
        return static_cast<U&>(value<T>());
    }
};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_VALUE_H

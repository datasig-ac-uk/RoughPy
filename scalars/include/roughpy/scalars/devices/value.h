//
// Created by sam on 3/29/24.
//

#ifndef ROUGHPY_DEVICES_VALUE_H
#define ROUGHPY_DEVICES_VALUE_H

#include <roughpy/core/traits.h>
#include <utility>

namespace rpy {
namespace devices {

class Reference
{
    void* p_val;

public:
    explicit Reference(void* val) : p_val(val) {}
    explicit Reference(const void* val) : p_val(const_cast<void*>(val)) {}

    template <typename T>
    add_const_t<T>& value() const
    {
        return *static_cast<add_const_t<T>*>(p_val);
    }

    template <typename T>
    T& value()
    {
        return *static_cast<T*>(p_val);
    }
};

template <typename T>
class TypedReference : public Reference
{
public:
    TypedReference(T& t)
        : Reference(const_cast<remove_cv_t<T>*>(std::addressof(t)))
    {}

    operator add_const_t<T>&() const { return *value<add_const_t<T>>(); }

    template <typename U = remove_cv_ref_t<T>>
    operator enable_if_t<is_convertible<T&, U&>::value, U&>()
    {
        return static_cast<U&>(value<T>());
    }
};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_VALUE_H

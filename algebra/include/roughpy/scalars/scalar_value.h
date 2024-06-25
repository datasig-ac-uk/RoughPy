//
// Created by sam on 24/06/24.
//

#ifndef ROUGHPY_SCALARS_SCALAR_VALUE_H
#define ROUGHPY_SCALARS_SCALAR_VALUE_H

#include "scalars_fwd.h"

#include <roughpy/devices/type.h>
#include <roughpy/devices/value.h>

namespace rpy {
namespace scalars {

class ScalarReference;
class ScalarConstReference;

class ScalarValue : public devices::Value
{
    ValueStorage p_fibre{};

public:
    operator ScalarConstReference() const noexcept;

    operator ScalarReference() noexcept;
};

namespace dtl {

template <typename T>
struct FibrePointerType {
    using type = const void*;
};

template <>
struct FibrePointerType<devices::Reference> {
    using type = void*;
};

template <typename Base>
struct ScalarReferenceBase : public Base
{
    using PtrType = typename FibrePointerType<Base>::type;
    PtrType p_fibre;

    ScalarReferenceBase(Base base, PtrType fibre)
        : Base(std::move(base)), p_fibre(fibre)
    {}


    const void* base_ptr() const noexcept
    {
        return Base::data();
    }

    template <typename T>
    enable_if_t<!is_void_v<T>, const T*> base_ptr() const noexcept {
        return Base::template data<T>();
    }

    const void* fibre_ptr() const noexcept
    {
        return p_fibre;
    }

    template <typename T>
    enable_if_t<!is_void_v<T>, const T*> fibre_ptr() const noexcept
    {
        return launder(static_cast<const T*>(p_fibre));
    }

    Base base() const noexcept { return *this; }

    Base fibre() const noexcept
    {
        return {p_fibre, Base::type()};
    }

};

}// namespace dtl

class ScalarConstReference
    : public dtl::ScalarReferenceBase<devices::ConstReference>
{
    using base_t = dtl::ScalarReferenceBase<devices::ConstReference>;
public:
    using base_t::base_t;
};

class ScalarReference : public dtl::ScalarReferenceBase<devices::Reference>
{
    using base_t = dtl::ScalarReferenceBase<devices::Reference>;
public:
    using base_t::base_t;


    operator ScalarConstReference() const noexcept
    {
        return ScalarConstReference(static_cast<const ConstReference&>(*this), this->p_fibre);
    }

    using base_t::base_ptr;

    void* base_ptr() noexcept
    {
        return this->data();
    }

    template <typename T>
    enable_if_t<!is_void_v<T>, T*> base_ptr() noexcept
    {
        return this->data<T>();
    }

    using base_t::fibre_ptr;

    void* fibre_ptr() noexcept
    {
        return this->p_fibre;
    }

    template <typename T>
    enable_if_t<!is_void_v<T>, T*> fibre_ptr() noexcept
    {
        return launder(static_cast<T*>(this->data()));
    }


    ConstReference base() const noexcept
    {
        return static_cast<const ConstReference&>(*this);
    }

    ConstReference fibre() const noexcept
    {
        return ConstReference{p_fibre, this->type()};
    }
};

inline ScalarValue::operator ScalarReference() noexcept
{
    return {static_cast<devices::Reference>(*this), p_fibre.data(&*type())};
}

inline ScalarValue::operator ScalarConstReference() const noexcept
{
    return {static_cast<devices::ConstReference>(*this), p_fibre.data(&*type())
    };
}

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_VALUE_H

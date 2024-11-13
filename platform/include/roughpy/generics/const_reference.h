//
// Created by sam on 13/11/24.
//

#ifndef CONST_REFERENCE_H
#define CONST_REFERENCE_H


#include <roughpy/core/traits.h>

#include "roughpy/platform/errors.h"
#include "roughpy/platform/roughpy_platform_export.h"

#include "roughpy/core/macros.h"
#include "type_ptr.h"
#include "type.h"

namespace rpy {
namespace generics {


class ConstReference;
class Reference;
class Value;




class ROUGHPY_PLATFORM_EXPORT ConstReference {
    TypePtr p_type;
    const void* p_data;

protected:

    struct without_null_check {};

    // ConstReference should not usually be constructed without
    // valid data, but internally this is a valid state. This
    // constructed is used internally for constructing (derived classes)
    // where the data pointer might be null. (See ConstPointer below.)
    // This causes the construct to skip the validity check.
    ConstReference(TypePtr type, const void* data, without_null_check)
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
    using const_reference = ConstReference;
    using reference = ConstReference;
    using pointer = ConstReference;


    ConstReference(TypePtr type, const void* p_data)
        : p_type(std::move(type)), p_data(p_data)
    {

    }


    RPY_NO_DISCARD
    bool is_valid() const noexcept
    {
        return static_cast<bool>(p_type) && p_data != nullptr;
    }

    RPY_NO_DISCARD
    bool fast_is_zero() const noexcept
    {
        return !is_valid();
    }



};

class ROUGHPY_PLATFORM_EXPORT ConstPointer : ConstReference
{

public:

    using value_type = Value;
    using const_reference = ConstReference;
    using reference = ConstReference;
    using pointer = ConstReference;

    explicit ConstPointer(TypePtr type, const void* data=nullptr)
        : ConstReference(std::move(type), data, without_null_check{})
    {
    }


    ConstReference operator*() const noexcept
    {
        RPY_DBG_ASSERT(is_valid())
        return static_cast<ConstReference>(*this);
    }

    const ConstReference* operator->() const noexcept
    {
        RPY_DBG_ASSERT(is_valid());
        return static_cast<const ConstReference*>(this);
    }


};



} // generics
} // rpy

#endif //CONST_REFERENCE_H

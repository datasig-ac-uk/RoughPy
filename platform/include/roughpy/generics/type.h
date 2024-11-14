//
// Created by sam on 13/11/24.
//

#ifndef ROUGHPY_GENERICS_TYPE_H
#define ROUGHPY_GENERICS_TYPE_H

#include <atomic>
#include <memory>
#include <typeinfo>

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/platform/roughpy_platform_export.h>

#include "type_ptr.h"

namespace rpy::generics {


class FromTrait;
class IntoTrait;



class ROUGHPY_PLATFORM_EXPORT Type
{
    mutable std::atomic_intptr_t m_rc;
    const std::type_info* p_type_info;

protected:

    explicit Type(const std::type_info* real_type_info)
        : m_rc(1), p_type_info(real_type_info)
    {}

    virtual ~Type();

    virtual void inc_ref() const noexcept;
    virtual bool dec_ref() const noexcept;

public:

    intptr_t ref_count() const noexcept;

    friend void intrusive_ptr_add_ref(const Type* value) noexcept
    {
        value->inc_ref();
    }

    friend void intrusive_ptr_release(const Type* value) noexcept
    {
        if (RPY_UNLIKELY(value->dec_ref())) {
            delete value;
        }
    }

public:

    RPY_NO_DISCARD
    const std::type_info& type_info() const noexcept { return *p_type_info; }

    RPY_NO_DISCARD
    virtual std::unique_ptr<const FromTrait> from(const Type& type) const noexcept;

    RPY_NO_DISCARD
    virtual std::unique_ptr<const IntoTrait> to(const Type& type) const noexcept;


    RPY_NO_DISCARD
    virtual std::unique_ptr<const FromTrait> from(const Type& type) const noexcept;

    RPY_NO_DISCARD
    virtual std::unique_ptr<const IntoTrait> to(const Type& type) const noexcept;





};





}

#endif //ROUGHPY_GENERICS_TYPE_H

//
// Created by sammorley on 10/11/24.
//

#ifndef ROUGHPY_GENERICS_TYPE_H
#define ROUGHPY_GENERICS_TYPE_H

#include <atomic>

#include "roughpy/core/macros.h"

#include "traits/builtin_trait.h"
#include "type_ptr.h"

#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy::generics {


class ROUGHPY_PLATFORM_EXPORT Type
{
    mutable std::atomic_intptr_t m_rc;

    using builtin_trait_ptr = std::unique_ptr<const BuiltinTraits>;

    std::array<builtin_trait_ptr, builtin_trait_count> m_builtin_traits;


protected:

    Type(std::array<builtin_trait_ptr, builtin_trait_count> builtin_traits)
        : m_rc(0), m_builtin_traits(std::move(builtin_traits))
    {

    }

    virtual ~Type() = default;

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


};

}

#endif //ROUGHPY_GENERICS_TYPE_H

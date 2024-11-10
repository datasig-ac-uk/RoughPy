//
// Created by sammorley on 10/11/24.
//


#include "roughpy/generics/type.h"

#include <roughpy/core/macros.h>


void rpy::generics::Type::inc_ref() const noexcept
{
    this->m_rc.fetch_add(1, std::memory_order_acq_rel);
}
bool rpy::generics::Type::dec_ref() const noexcept
{
    RPY_DBG_ASSERT(m_rc.load() > 1);
    if (this->m_rc.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        return true;
    }
    return false;
}
intptr_t rpy::generics::Type::ref_count() const noexcept
{
    return m_rc.load();
}

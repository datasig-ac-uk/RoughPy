//
// Created by sam on 16/02/24.
//

#include "basis_key.h"

using namespace rpy;
using namespace rpy::algebra;


BasisKeyInterface::~BasisKeyInterface() = default;

uint32_t BasisKeyInterface::inc_ref() const noexcept
{
    auto count
            = m_ref_count.fetch_add(1, std::memory_order::memory_order_acquire);
    return count + 1;
}

uint32_t BasisKeyInterface::dec_ref() const noexcept
{
    auto count
            = m_ref_count.fetch_sub(1, std::memory_order::memory_order_release);
    RPY_DBG_ASSERT(count > 0);
    return count - 1;
}

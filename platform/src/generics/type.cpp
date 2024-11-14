//
// Created by sam on 14/11/24.
//

#include "roughpy/generics/type.h"

#include <memory>



#include "into_from.h"

using namespace rpy;
using namespace rpy::generics;

Type::~Type() = default;

void Type::inc_ref() const noexcept
{
    this->m_rc.fetch_add(1, std::memory_order_relaxed);
}
bool Type::dec_ref() const noexcept
{
    auto old = this->m_rc.fetch_sub(1, std::memory_order_acq_rel);
    return old == 1;
}

intptr_t Type::ref_count() const noexcept
{
    return this->m_rc.load(std::memory_order_acquire);
}

std::unique_ptr<const FromTrait> Type::from(const Type& type) const noexcept
{
return nullptr;
}

std::unique_ptr<const IntoTrait> Type::to(const Type& type) const noexcept
{
    if (auto p_from = from(type)) {
        return std::make_unique<IntoFrom>(std::move(p_from));
    }

    return nullptr;
}

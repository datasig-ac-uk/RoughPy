//
// Created by sam on 14/11/24.
//

#include "roughpy/generics/type.h"

#include <memory>

#include "roughpy/generics/conversion_trait.h"

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

std::unique_ptr<const ConversionTrait>
Type::convert_to(const Type& type, bool try_pass) const noexcept
{
    if (try_pass) { return type.convert_to(*this, false); }
    return nullptr;
}

const BuiltinTrait* Type::get_builtin_trait(BuiltinTraitID id) const noexcept
{
    return nullptr;
}
const Trait* Type::get_trait(string_view id) const noexcept { return nullptr; }

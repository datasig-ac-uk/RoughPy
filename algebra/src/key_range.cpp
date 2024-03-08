//
// Created by sam on 1/31/24.
//


#include "basis.h"

using namespace rpy;
using namespace rpy::algebra;


KeyRange::KeyRange() = default;

KeyRange::KeyRange(KeyIteratorState* state) noexcept : p_state(state)
{
    RPY_DBG_ASSERT(state != nullptr);
}

KeyRange::KeyRange(KeyRange&& other) noexcept
    : p_state(other.p_state)
{
    other.p_state = nullptr;
}

KeyRange::~KeyRange()
{
    if (p_state != nullptr) {
        delete p_state;
        p_state = nullptr;
    }
}

KeyRange& KeyRange::operator=(KeyRange&& other) noexcept
{
    if (&other != this) {
        this->~KeyRange();
        p_state = other.p_state;
        other.p_state = nullptr;
    }
    return *this;
}

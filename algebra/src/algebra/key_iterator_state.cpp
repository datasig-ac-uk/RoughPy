//
// Created by sam on 3/15/24.
//

#include "basis.h"

#include <roughpy/platform/alloc.h>

using namespace rpy;
using namespace rpy::algebra;

void* KeyIteratorState::operator new(std::size_t count)
{
    return platform::alloc_small(count);
}

void KeyIteratorState::operator delete(void* ptr, std::size_t count)
{
    platform::free_small(ptr, count);
}

KeyIteratorState::~KeyIteratorState() = default;

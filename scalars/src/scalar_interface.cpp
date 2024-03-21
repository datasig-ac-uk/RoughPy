//
// Created by sam on 13/11/23.
//

#include "scalar_interface.h"

#include <roughpy/platform/alloc.h>

using namespace rpy;
using namespace rpy::scalars;

void* ScalarInterface::operator new(std::size_t count)
{
    return platform::alloc_small(count);
}
void ScalarInterface::operator delete(void* ptr, std::size_t count)
{
    platform::free_small(ptr, count);
}

ScalarInterface::~ScalarInterface() = default;
void ScalarInterface::add_inplace(const Scalar& other) {}
void ScalarInterface::sub_inplace(const Scalar& other) {}
void ScalarInterface::mul_inplace(const Scalar& other) {}
void ScalarInterface::div_inplace(const Scalar& other) {}

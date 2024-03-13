//
// Created by sam on 13/11/23.
//

#include "scalar_interface.h"

using namespace rpy;
using namespace rpy::scalars;

void* ScalarInterface::operator new(
        std::size_t count,
        std::align_val_t alignment
)
{
    return platform::alloc_small(count, alignment);
}
void ScalarInterface::operator delete(
        void* ptr,
        std::size_t count,
        std::align_val_t alignment
)
{
    platform::free_small(ptr, count, alignment);
}

ScalarInterface::~ScalarInterface() = default;
void ScalarInterface::add_inplace(const Scalar& other) {}
void ScalarInterface::sub_inplace(const Scalar& other) {}
void ScalarInterface::mul_inplace(const Scalar& other) {}
void ScalarInterface::div_inplace(const Scalar& other) {}

#ifndef ROUGHPY_CORE_ALLOC_H_
#define ROUGHPY_CORE_ALLOC_H_

#include <boost/align/aligned_alloc.hpp>
#include <cstdlib>

#include "traits.h"

namespace rpy {

using std::calloc;
using std::free;
using std::malloc;
using std::realloc;

// using std::aligned_alloc;

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;


template <typename T, typename... Args>
constexpr void construct_inplace(T* dst, Args&&... args)
    noexcept(is_nothrow_constructible<T, Args...>::value)
{
    ::new (static_cast<void*>(dst)) T(std::forward<Args>(args)...);
}


}// namespace rpy

#endif// ROUGHPY_CORE_ALLOC_H_

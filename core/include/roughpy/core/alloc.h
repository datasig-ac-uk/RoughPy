#ifndef ROUGHPY_CORE_ALLOC_H_
#define ROUGHPY_CORE_ALLOC_H_

#include <boost/align/aligned_alloc.hpp>

#include "traits.h"

#include <cstdlib>
#include <memory>
#include <new>

namespace rpy {

using std::calloc;
using std::free;
using std::malloc;
using std::realloc;

// using std::aligned_alloc;

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

/**
 * \brief Constructs an object in place at the given destination address.
 *
 * \tparam T The type of the object to construct.
 * \tparam Args The types of the arguments to pass to the constructor.
 * \param dst Pointer to the memory where the object will be constructed.
 * \param args Arguments to pass to the constructor.
 *
 * \note This function assumes that the memory at 'dst' is uninitialized and can
 * be used for object construction.
 *
 * \par Example
 *
 * \code{.cpp}
 * int* p = (int*)malloc(sizeof(int)); // Allocate memory
 * construct_inplace(p, 5); // Construct an 'int' object with the value '5'
 * int value = *p; // Access the constructed object
 * free(p); // Clean up the allocated memory
 * \endcode
 *
 * \remark This function is noexcept if the constructor of type 'T' with
 * 'Args...' is noexcept.
 */
template <typename T, typename... Args>
constexpr void construct_inplace(
        T* dst,
        Args&&... args
) noexcept(is_nothrow_constructible_v<T, Args...>)
{
    ::new (static_cast<void*>(dst)) T{std::forward<Args>(args)...};
}

template <typename T>
using Allocator = std::allocator<T>;

}// namespace rpy

#endif// ROUGHPY_CORE_ALLOC_H_

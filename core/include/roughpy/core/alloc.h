#ifndef ROUGHPY_CORE_ALLOC_H_
#define ROUGHPY_CORE_ALLOC_H_

#include <boost/align/aligned_alloc.hpp>
#include <cstdlib>

namespace rpy {

using std::calloc;
using std::free;
using std::malloc;
using std::realloc;

// using std::aligned_alloc;

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;

}// namespace rpy

#endif// ROUGHPY_CORE_ALLOC_H_

#ifndef ROUGHPY_CORE_ALLOC_H_
#define ROUGHPY_CORE_ALLOC_H_

#include <cstdlib>
#include <boost/align/aligned_alloc.hpp>


namespace rpy {

using std::calloc;
using std::malloc;
using std::realloc;
using std::free;


//using std::aligned_alloc;

using boost::alignment::aligned_alloc;
using boost::alignment::aligned_free;


}


#endif // ROUGHPY_CORE_ALLOC_H_

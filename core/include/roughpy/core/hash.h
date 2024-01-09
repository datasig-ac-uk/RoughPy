//
// Created by sam on 09/01/24.
//

#ifndef ROUGHPY_CORE_HASH_H
#define ROUGHPY_CORE_HASH_H

#include "types.h"

#include <boost/container_hash/hash.hpp>

namespace rpy {


using hash_t = std::size_t;

template <typename T>
using Hash = boost::hash<T>;

using boost::hash_combine;

}


#endif// ROUGHPY_CORE_HASH_H

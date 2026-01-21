//
// Created by sam on 09/01/24.
//

#ifndef ROUGHPY_CORE_HASH_H
#define ROUGHPY_CORE_HASH_H

/*
 * Ideally we'd use an off-the-shelf hash like std::hash or boost::hash (from
 * container_hash) but in both cases we have a problem. The former does not
 * define hash for many stl containers like pair or tuple, and container_hash
 * requires mp11 for some reason. The biggest problem with std::hash is how
 * irritating it is to extend with new types. Boost's solution is far nicer
 * and uses ADL to find a special function hash_value.
 *
 * Reluctantly, for now at least, we will continue to use boost container_hash.
 * We might seriously consider replacing this in due course if we decide to get
 * rid of boost mp11.
 */
#include <boost/container_hash/hash.hpp>

namespace rpy {

using hash_t = std::size_t;

using boost::hash_value;
using boost::hash_combine;


template <typename T=void>
struct Hash : public boost::hash<T> {};

template <>
struct Hash<void>
{

    template <typename T>
    hash_t operator()(const T& value) noexcept(noexcept(boost::hash<T>{}(value)))
    {
        return boost::hash<T>{}(value);
    }

};



}


#endif// ROUGHPY_CORE_HASH_H

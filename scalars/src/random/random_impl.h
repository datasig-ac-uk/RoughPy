//
// Created by sam on 12/05/23.
//

#ifndef ROUGHPY_RANDOM_IMPL_H
#define ROUGHPY_RANDOM_IMPL_H

#include <pcg_random.hpp>
#include <random>

namespace rpy {
namespace scalars {
namespace dtl {

template <typename>
struct rng_type_getter;

template <>
struct rng_type_getter<std::mt19937_64> {
    static const char* name;
};

template <>
struct rng_type_getter<pcg64> {
    static const char* name;
};

}// namespace dtl
}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_RANDOM_IMPL_H

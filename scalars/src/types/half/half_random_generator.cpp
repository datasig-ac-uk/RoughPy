//
// Created by sam on 11/17/23.
//

#include "half_random_generator.h"

namespace rpy {
namespace scalars {

template class StandardRandomGenerator<Half, std::mt19937_64>;
template class StandardRandomGenerator<Half, pcg64>;

} // scalars
} // rpy

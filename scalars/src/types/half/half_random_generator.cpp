//
// Created by sam on 11/17/23.
//

#include "half_random_generator.h"

namespace rpy {
namespace scalars {

template class StandardRandomGenerator<half, std::mt19937_64>;
template class StandardRandomGenerator<half, pcg64>;

} // scalars
} // rpy

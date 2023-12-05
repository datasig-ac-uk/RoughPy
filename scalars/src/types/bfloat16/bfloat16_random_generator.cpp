//
// Created by sam on 11/17/23.
//

#include "bfloat16_random_generator.h"

namespace rpy {
namespace scalars {

template class StandardRandomGenerator<bfloat16, std::mt19937_64>;
template class StandardRandomGenerator<bfloat16, pcg64>;

} // scalars
} // rpy

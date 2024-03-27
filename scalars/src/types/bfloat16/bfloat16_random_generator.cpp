//
// Created by sam on 11/17/23.
//

#include "bfloat16_random_generator.h"
#include "scalar_implementations/bfloat.h"

namespace rpy {
namespace scalars {

template class StandardRandomGenerator<BFloat16, std::mt19937_64>;
template class StandardRandomGenerator<BFloat16, pcg64>;

} // scalars
} // rpy

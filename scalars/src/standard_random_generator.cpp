//
// Created by user on 20/03/23.
//

#include "standard_random_generator.h"

namespace rpy {
namespace scalars {

template class StandardRandomGenerator<float, std::mt19937_64>;
template class StandardRandomGenerator<float, pcg64>;
template class StandardRandomGenerator<double, std::mt19937_64>;
template class StandardRandomGenerator<double, pcg64>;


}// namespace scalars
}// namespace rpy

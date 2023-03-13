//
// Created by user on 27/02/23.
//

#include "scalar_interface.h"

#include "scalar.h"

using namespace rpy::scalars;

Scalar ScalarInterface::uminus() const {
    return Scalar();
}
bool ScalarInterface::equals(const Scalar &other) const noexcept {
    return false;
}
std::ostream &ScalarInterface::print(std::ostream &os) const {
    return os;
}

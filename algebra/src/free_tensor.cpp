//
// Created by user on 05/03/23.
//

#include "free_tensor.h"

using namespace rpy::algebra;

namespace rpy {
namespace algebra {
template class AlgebraInterface<FreeTensor>;
template class AlgebraBase<FreeTensorInterface, FreeTensorImplementation>;
}// namespace algebra
}// namespace rpy

FreeTensor FreeTensor::exp() const {
    if (p_impl) {
        return p_impl->exp();
    }
    return {};
}
FreeTensor FreeTensor::log() const {
    if (p_impl) {
        return p_impl->log();
    }
    return {};
}
FreeTensor FreeTensor::inverse() const {
    if (p_impl) {
        return p_impl->inverse();
    }
    return {};
}
FreeTensor FreeTensor::antipode() const {
    if (p_impl) {
        return p_impl->antipode();
    }
    return {};
}
FreeTensor& FreeTensor::fmexp(const FreeTensor &other) {
    if (p_impl && !is_equivalent_to_zero(other)) {
       p_impl->fmexp(*other);
    }
    return *this;
}

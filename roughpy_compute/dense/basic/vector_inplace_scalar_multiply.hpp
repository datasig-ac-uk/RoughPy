#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_INPLACE_SCALAR_MULTIPLY_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_INPLACE_SCALAR_MULTIPLY_HPP

#include "roughpy_compute/dense/views.hpp"

namespace rpy::compute::basic {
inline namespace v1 {


template <typename S>
void vector_inplace_scalar_multiply(DenseVectorView<S*> arg, S const& scalar)
{
    using Size = typename DenseVectorView<S*>::Size;

    for (Size i=0; i < arg.size(); ++i) {
        arg[i] *= scalar;
    }
}

} // version namespace
} // namespace rpy::compute::basic

#endif //ROUGHPY_COMPUTE_DENSE_BASIC_VECTOR_INPLACE_SCALAR_MULTIPLY_HPP

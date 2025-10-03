#ifndef ROUGHPY_COMPUTE_COMMON_SCALARS_HPP
#define ROUGHPY_COMPUTE_COMMON_SCALARS_HPP


namespace rpy::compute::scalars {


template <typename T>
struct ScalarTraits
{
    using Scalar = T;
    using Rational = T;
    using Real = T;

    static const T zero;
};

template <typename T>
const T ScalarTraits<T>::zero = T(0);


} // namespace rpy::compute::scalars

#endif //ROUGHPY_COMPUTE_COMMON_SCALARS_HPP
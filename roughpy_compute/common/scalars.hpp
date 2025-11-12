#ifndef ROUGHPY_COMPUTE_COMMON_SCALARS_HPP
#define ROUGHPY_COMPUTE_COMMON_SCALARS_HPP


namespace rpy::compute::scalars {


template <typename T>
struct Traits
{
    using Scalar = T;
    using Rational = T;
    using Real = T;
    using error_type = void;

    using reference = T&;
    using const_reference = const T&;






};



} // namespace rpy::compute::scalars

#endif //ROUGHPY_COMPUTE_COMMON_SCALARS_HPP
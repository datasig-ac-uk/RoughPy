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



    static constexpr T zero() noexcept
    {
        return T(0);
    }

    static constexpr T one() noexcept
    {
        return T(1);
    }


};



} // namespace rpy::compute::scalars

#endif //ROUGHPY_COMPUTE_COMMON_SCALARS_HPP
#ifndef ROUGHPY_COMPUTE_COMMON_BASIS_H
#define ROUGHPY_COMPUTE_COMMON_BASIS_H


#include "architecture.hpp"

namespace rpy::compute {

template <typename Architecture_>
struct BasisBase
{
    using Size = typename Architecture_::Size;
    using Degree = typename Architecture_::Degree;

    Size const* degree_begin;
    Degree width;
    Degree depth;
};

template <typename Architecture_=NativeArchitecture>
struct TensorBasis : BasisBase<Architecture_>
{
};



template <typename Architecture_=NativeArchitecture>
struct LieBasis : BasisBase<Architecture_>
{
    using Base = BasisBase<Architecture_>;
    using typename Base::Size;;

    Size const* data;
};


} // namespace rpy::compute

#endif // ROUGHPY_COMPUTE_COMMON_BASIS_H

#ifndef ROUGHPY_COMPUTE_COMMON_BASIS_HPP
#define ROUGHPY_COMPUTE_COMMON_BASIS_HPP

#include <algorithm>

#include "architecture.hpp"

namespace rpy::compute {

template <typename Architecture_>
struct BasisBase
{
    using Architecture = Architecture_;
    using Size = typename Architecture::Size;
    using Index = typename Architecture::Index;
    using Degree = typename Architecture::Degree;

    Size const* degree_begin; // size depth + 2
    Degree width;
    Degree depth;

    Size size() const noexcept { return degree_begin[depth+1]; }
};

template <typename Architecture_=NativeArchitecture>
struct TensorBasis : BasisBase<Architecture_>
{

    [[nodiscard]]
    constexpr TensorBasis truncate(Degree new_depth) const noexcept
    {
        return {
            this->degree_begin,
            this->width,
            std::min(this->depth, new_depth),
        };
    }
};



template <typename Architecture_=NativeArchitecture>
struct LieBasis : BasisBase<Architecture_>
{
    using Base = BasisBase<Architecture_>;
    using typename Base::Size;;

    Size const* data;


    [[nodiscard]]
    constexpr LieBasis truncate(Degree new_depth) const noexcept
    {
        return {
            this->degree_begin,
            this->width,
            std::min(this->depth, new_depth),
            this->data
        }
    }
};


} // namespace rpy::compute

#endif // ROUGHPY_COMPUTE_COMMON_BASIS_HPP

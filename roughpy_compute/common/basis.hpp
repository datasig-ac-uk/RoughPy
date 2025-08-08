#ifndef ROUGHPY_COMPUTE_COMMON_BASIS_HPP
#define ROUGHPY_COMPUTE_COMMON_BASIS_HPP

#include <algorithm>

#include "architecture.hpp"

namespace rpy::compute {

struct BasisBase
{
    using Size = std::size_t;
    using Index = std::ptrdiff_t;
    using Degree = int32_t;

    Index const* degree_begin; // size depth + 2
    Degree width;
    Degree depth;

    Index size() const noexcept { return degree_begin[depth+1]; }
};

struct TensorBasis : BasisBase
{
    using Base = BasisBase;
    using typename Base::Size;
    using typename Base::Degree;

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



struct LieBasis : BasisBase
{
    using Base = BasisBase;
    using typename Base::Index;
    using typename Base::Degree;

    Index const* data;


    [[nodiscard]]
    constexpr LieBasis truncate(Degree new_depth) const noexcept
    {
        return {
                {
                    this->degree_begin,
                   this->width,
                   std::min(this->depth, new_depth)
                },
            this->data
        };
    }
};


} // namespace rpy::compute

#endif // ROUGHPY_COMPUTE_COMMON_BASIS_HPP

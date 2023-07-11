#ifndef ROUGHPY_INTERVALS_PARTITION_H_
#define ROUGHPY_INTERVALS_PARTITION_H_

#include "real_interval.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>
#include <roughpy/platform/serialization.h>

namespace rpy {
namespace intervals {

class RPY_EXPORT Partition : public RealInterval
{
public:
    using intermediates_t = std::vector<param_t>;

private:
    intermediates_t m_intermediate_points;

public:
    explicit Partition(RealInterval base);
    explicit Partition(RealInterval base, Slice<param_t> intermediate_points);

    Partition(RealInterval base, std::vector<param_t>&& intermediates)
        : RealInterval(std::move(base)),
          m_intermediate_points(std::move(intermediates))
    {}

    RPY_NO_DISCARD
    Partition refine_midpoints() const;

    RPY_NO_DISCARD
    param_t mesh() const noexcept;

    RPY_NO_DISCARD
    dimn_t size() const noexcept { return 1 + m_intermediate_points.size(); }

    RPY_NO_DISCARD
    RealInterval operator[](dimn_t i) const;

    const intermediates_t& intermediates() const noexcept
    {
        return m_intermediate_points;
    }

    void insert_intermediate(param_t new_intermediate);

    RPY_NO_DISCARD
    Partition merge(const Partition& other) const;
};

}// namespace intervals
}// namespace rpy

#endif// ROUGHPY_INTERVALS_PARTITION_H_

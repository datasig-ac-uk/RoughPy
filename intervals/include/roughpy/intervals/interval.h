#ifndef ROUGHPY_INTERVALS_INTERVAL_H_
#define ROUGHPY_INTERVALS_INTERVAL_H_

#include "roughpy_intervals_export.h"

#include <cassert>
#include <cstdint>
#include <iosfwd>

namespace rpy { namespace intervals {

using param_t = double;
using dyadic_multiplier_t = int;
using dyadic_depth_t = int;



enum class IntervalType {
    Clopen,
    Opencl
};


class ROUGHPY_INTERVALS_EXPORT Interval {
protected:
    IntervalType m_interval_type = IntervalType::Clopen;

public:

    Interval() = default;

    explicit Interval(IntervalType itype) : m_interval_type(itype)
    {}

    virtual ~Interval() = default;

    IntervalType type() const noexcept { return m_interval_type; }

    virtual param_t inf() const = 0;
    virtual param_t sup() const = 0;

    virtual param_t included_end() const;
    virtual param_t excluded_end() const;

    virtual bool contains(param_t arg) const noexcept;
    virtual bool is_associated(const Interval& arg) const noexcept;
    virtual bool contains(const Interval& arg) const noexcept;
    virtual bool intersects_with(const Interval& arg) const noexcept;

    virtual bool operator==(const Interval& other) const;
    virtual bool operator!=(const Interval& other) const;

};


ROUGHPY_INTERVALS_EXPORT
std::ostream& operator<<(std::ostream& os, const Interval& interval);


}}

#endif // ROUGHPY_INTERVALS_INTERVAL_H_

#ifndef ROUGHPY_INTERVALS_REAL_INTERVAL_H_
#define ROUGHPY_INTERVALS_REAL_INTERVAL_H_

#include "interval.h"

#include <utility>

namespace rpy { namespace intervals {

class ROUGHPY_INTERVALS_EXPORT RealInterval : public Interval {
    param_t m_inf = 0.0;
    param_t m_sup = 1.0;

public:

    RealInterval(param_t inf, param_t sup,
                 IntervalType itype=IntervalType::Clopen)
        : Interval(itype), m_inf(inf), m_sup(sup)
    {
        if (m_inf < m_sup) {
            std::swap(m_inf, m_sup);
        }
    }

    explicit RealInterval(const Interval& interval)
        : Interval(interval.type()),
          m_inf(interval.inf()),
          m_sup(interval.sup())
    {}

    explicit RealInterval(const Interval& interval, IntervalType itype)
        : Interval(itype), m_inf(interval.inf()), m_sup(interval.sup())
    {}


    param_t inf() const override { return m_inf; }
    param_t sup() const override { return m_sup; }

    bool contains(const Interval &arg) const noexcept override;

};


}}


#endif // ROUGHPY_INTERVALS_REAL_INTERVAL_H_

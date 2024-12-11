//
// Created by sammorley on 03/12/24.
//

#ifndef ROUGHPY_STREAMS_ARRIVAL_STREAM_H
#define ROUGHPY_STREAMS_ARRIVAL_STREAM_H

#include <functional>

#include <boost/container/flat_map.hpp>

#include "roughpy/intervals/interval.h"
#include "roughpy/intervals/dyadic_interval.h"
#include "roughpy/intervals/real_interval.h"

#include <roughpy/algebra/lie.h>

#include "stream_base.h"
#include "roughpy_streams_export.h"

namespace rpy::streams {





class ROUGHPY_STREAMS_EXPORT ArrivalStream : public StreamInterface {
    using DyadicInterval = intervals::DyadicInterval;
    using Interval = intervals::Interval;
    using Lie = algebra::Lie;

    using virtual_func = std::function<Lie(param_t)>;

    using virtual_map_type = boost::container::flat_map<param_t, virtual_func>;
    using arrivals_map_type = boost::container::flat_map<param_t, Lie>;

    arrivals_map_type m_arrivals;
public:

    RPY_NO_DISCARD bool
    empty(const intervals::Interval& interval) const noexcept override;

protected:

    RPY_NO_DISCARD algebra::Lie log_signature_impl(
        const intervals::Interval& interval,
        const algebra::Context& ctx) const override;

};



}


#endif //ROUGHPY_STREAMS_ARRIVAL_STREAM_H

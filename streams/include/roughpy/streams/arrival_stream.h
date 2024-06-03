//
// Created by sam on 5/31/24.
//

#ifndef ROUGHPY_STREAMS_ARRIVAL_STREAM_H
#define ROUGHPY_STREAMS_ARRIVAL_STREAM_H

#include "stream_base.h"

#include <roughpy/core/containers/map.h>

namespace rpy { namespace streams {


class ROUGHPY_STREAMS_EXPORT ArrivalsStreams : public StreamInterface {
    using intervals::RealInterval;
    using intervals::Interval;

    algebra::BasisKey m_pen_on_key;
    container::Map<param_t, algebra::Lie> m_arrivals;

    RPY_NO_DISCARD algebra::FreeTensor
    sig_to_intervalt(const Interval& interval, const algebra::Context& ctx) const;

    RPY_NO_DISCARD algebra::FreeTensor
    pen_on_sig(const algebra::Context& ctx) const;

    RPY_NO_DISCARD algebra::FreeTensor
    sig_of_interval(const Interval& interval,
         const algebra::Context& ctx) const;
public:



    RPY_NO_DISCARD algebra::Lie log_signature_impl(
        const intervals::Interval& interval,
        const algebra::Context& ctx
    ) const override;


};


}}

#endif //ROUGHPY_STREAMS_ARRIVAL_STREAM_H

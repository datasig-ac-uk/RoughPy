//
// Created by sammorley on 03/12/24.
//


#include "arrival_stream.h"

#include <roughpy/scalars/scalar_type.h>
#include <roughpy/algebra/context.h>
#include <roughpy/algebra/free_tensor.h>

using namespace rpy;
using namespace rpy::streams;


bool ArrivalStream::empty(const intervals::Interval& interval) const noexcept
{
    auto lower_bound = m_arrivals.lower_bound(interval.inf());
    auto upper_bound = m_arrivals.upper_bound(interval.sup());

    return upper_bound == lower_bound;
}

algebra::Lie ArrivalStream::log_signature_impl(
    const intervals::Interval& interval,
    const algebra::Context& ctx) const
{
    auto lower_bound = m_arrivals.lower_bound(interval.inf());
    auto upper_bound = m_arrivals.upper_bound(interval.sup());

    auto result  = unit_tensor();
    for (auto it=lower_bound; it != upper_bound; ++it) {
        result.fmexp(ctx.lie_to_tensor(it->second));
    }

    return ctx.tensor_to_lie(result.log());
}
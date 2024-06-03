//
// Created by sam on 5/31/24.
//


#include "arrivals_stream.h"

using namespace rpy;
using namespace rpy::streams;

algebra::FreeTensor sig_to_interval(const intervals::RealInterval& interval, const algebra::Context& ctx) const
{

    return algebra::FreeTensor(ctx);
}

algebra::FreeTensor pen_on_sig(const algebra::Context& ctx) const {
    RPY_CHECK(ctx.width() == metedata.().width);
    return algebra::FreeTensor(m_pen_on_key, ctx.scalar_type()->one(), ctx).exp()
}

algebra::FreeTensor sig_of_interval(const intervals::Interval& interval, const algebra::Context& ctx) const
{
    auto lower_bound = m_arrivals.lower_bound(interval.inf());
    auto upper_bound = m_arrivals.lower_bound(interval.sup());

    algebra::FreeTensor result(ctx.scalar_type()->one(), ctx);
    for (auto it=lower_bound; it != upper_bound; ++it) {
        result *= ctx.lie_to_tensor(it->second).exp();
    }
    return result;
}

algebra::Lie
ArrivalsStream::log_signature_impl(const intervals::Interval& interval,
    const algebra::Context& ctx) const
{
    auto sig = sig_to_interval(interval, ctx) * pen_on_sig(ctx) * sig_of_interval(interval, ctx);
    return ctx.tensor_to_lie(sig.log());
}

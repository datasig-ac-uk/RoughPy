//
// Created by sam on 05/12/24.
//

#include "tensor_valued_stream.h"


using namespace rpy;
using namespace rpy::streams;


TensorValuedStream::TensorValuedStream(intervals::RealInterval domain,
                                       std::shared_ptr<const StreamInterface>
                                       increment_stream,
                                       std::shared_ptr<update_fn> updater,
                                       StreamValue initial_value,
                                       algebra::context_pointer ctx
)
    : m_domain(std::move(domain)),
      p_increment_stream(std::move(increment_stream)),
      p_updater(std::move(updater)),
      m_initial_value(std::move(initial_value)),
      p_ctx(std::move(ctx)) {}

bool TensorValuedStream::empty(
    const intervals::Interval& interval) const noexcept
{
    if (!m_domain.intersects_with(interval)) {
        return true;
    }
    return p_increment_stream->empty(intersection(m_domain, interval));
}

algebra::Lie TensorValuedStream::log_signature_impl(
    const intervals::Interval& interval,
    const algebra::Context& ctx) const
{
    return p_increment_stream->log_signature(interval, ctx);
}

const intervals::RealInterval& TensorValuedStream::domain() const noexcept
{
    return m_domain;
}

std::shared_ptr<const ValueStream> TensorValuedStream::query(
    const intervals::Interval& interval) const
{
    RPY_CHECK(m_domain.intersects_with(interval));

    auto query_interval = intersection(m_domain, interval);

    auto inf = query_interval.inf();
    return std::make_shared<TensorValuedStream>(std::move(query_interval),
                                                p_increment_stream,
                                                p_updater,
                                                value_at(inf),
                                                p_ctx);
}

std::shared_ptr<const StreamInterface> TensorValuedStream::
increment_stream() const noexcept { return p_increment_stream; }

ValueStream::StreamValue TensorValuedStream::value_at(param_t param) const
{
    if (param < m_domain.inf() || param > m_domain.sup()) {
        RPY_THROW(std::invalid_argument,
                  "param is not in the domain of this stream");
    }
    intervals::RealInterval interval(m_domain.inf(), param);
    return (*p_updater)(initial_value(),
                        p_increment_stream->log_signature(interval, *p_ctx));
}

ValueStream::StreamValue TensorValuedStream::initial_value() const
{
    return m_initial_value;
}

ValueStream::StreamValue TensorValuedStream::terminal_value() const
{
    return (*p_updater)(m_initial_value,
                        p_increment_stream->log_signature(m_domain, *p_ctx));
}
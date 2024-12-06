//
// Created by sam on 05/12/24.
//

#include "tensor_valued_stream.h"
#include "roughpy/streams/value_stream.h"


#include "roughpy/algebra/algebra_base.h"
#include "roughpy/algebra/free_tensor.h"
#include "roughpy/algebra/lie.h"


using namespace rpy;
using namespace rpy::streams;





TensorValuedStream::TensorValuedStream(intervals::RealInterval domain,
                                       std::shared_ptr<const StreamInterface>
                                       increment_stream,
                                       StreamValue initial_value,
                                       algebra::context_pointer ctx
)
    : m_domain(std::move(domain)),
      p_increment_stream(std::move(increment_stream)),
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

std::shared_ptr<const ValueStream<algebra::FreeTensor>> TensorValuedStream::query(
    const intervals::Interval& interval) const
{
    RPY_CHECK(m_domain.intersects_with(interval));

    auto query_interval = intersection(m_domain, interval);

    auto inf = query_interval.inf();
    return std::make_shared<TensorValuedStream>(std::move(query_interval),
                                                p_increment_stream,
                                                value_at(inf),
                                                p_ctx);
}

std::shared_ptr<const StreamInterface> TensorValuedStream::
increment_stream() const noexcept { return p_increment_stream; }

TensorValuedStream::StreamValue TensorValuedStream::value_at(param_t param) const
{
    if (param < m_domain.inf() || param > m_domain.sup()) {
        RPY_THROW(std::invalid_argument,
                  "param is not in the domain of this stream");
    }
    const intervals::RealInterval interval(m_domain.inf(), param);

    auto sig = p_increment_stream->signature(interval, *p_ctx);
    return sig.mul(m_initial_value);
}

TensorValuedStream::StreamValue TensorValuedStream::initial_value() const
{
    return m_initial_value;
}

TensorValuedStream::StreamValue TensorValuedStream::terminal_value() const
{
    auto sig = p_increment_stream->signature(m_domain, *p_ctx);
    return sig.mul(m_initial_value);
}
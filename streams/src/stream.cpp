// Copyright (c) 2023 RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <roughpy/streams/stream.h>

#include <roughpy/intervals/partition.h>
#include <roughpy/streams/piecewise_abelian_stream.h>

using namespace rpy;
using namespace rpy::streams;

bool Stream::check_support_and_trim(Stream::RealInterval& domain) const noexcept
{
    if (domain.sup() < m_support.inf() || domain.inf() > m_support.sup()) {
        // The intervals don't intersect, return false
        return false;
    }

    // Intervals do intersect, trim domain so it is a subset of m_support
    domain = RealInterval(
            std::max(domain.inf(), m_support.inf()),
            std::min(domain.sup(), m_support.sup()),
            domain.type()
    );

    return true;
}

inline optional<pair<Stream::RealInterval, resolution_t>>
Stream::refine_interval(
    const Interval& original_query
) const
{
    auto query = schema().adjust_interval(original_query);

    if (!check_support_and_trim(query)) { return {}; }
    auto length = query.sup() - query.inf();
    if (length == 0.0) { return {}; }

    auto resolution = std::max(
            metadata().default_resolution,
            param_to_resolution(length) + 2
            );

    return {{query, resolution}};
}

void Stream::restrict_to(const Stream::Interval& interval)
{
    if (p_impl) {
        m_support = p_impl->schema().adjust_interval(interval);
    } else {
        m_support = RealInterval(interval);
    }
}

Stream Stream::restrict(const Stream::Interval& interval) const
{
    RealInterval support(interval);
    if (p_impl) { support = p_impl->schema().adjust_interval(interval); }

    return {p_impl, support};
}

const algebra::Context& rpy::streams::Stream::get_default_context() const
{
    RPY_CHECK(p_impl);
    return *p_impl->metadata().default_context;
}
const rpy::streams::StreamMetadata& rpy::streams::Stream::metadata() const
{
    RPY_CHECK(p_impl);
    return p_impl->metadata();
}

const StreamSchema& Stream::schema() const
{
    RPY_CHECK(p_impl);
    return p_impl->schema();
}

inline Stream::Lie Stream::log_signature_impl(
        const Stream::Interval& interval,
        resolution_t resolution,
        const Stream::Context& ctx
) const
{
    const auto& md = metadata();
    auto dyadic_queries = intervals::to_dyadic_intervals(interval, resolution);
    std::vector<Lie> results;
    results.reserve(dyadic_queries.size());
    for (const auto& di : dyadic_queries) {
        results.push_back(p_impl->log_signature(di, resolution, ctx));
        if (results.back().is_zero()) { results.pop_back(); }
    }

    return ctx.cbh(results, md.cached_vector_type);
}

Stream::Lie Stream::log_signature(
        const Stream::Interval& interval,
        const Stream::Context& ctx
) const
{
    auto query_params = refine_interval(interval);
    if (!query_params) {
        return zero_lie(ctx);
    }
    return log_signature_impl(query_params->first, query_params->second, ctx);
}

rpy::streams::Stream::Lie rpy::streams::Stream::log_signature(
        const rpy::streams::Stream::Interval& interval,
        rpy::resolution_t resolution
) const
{
    const auto& md = metadata();
    return log_signature(interval, resolution, *md.default_context);
}
rpy::streams::Stream::Lie rpy::streams::Stream::log_signature(
        const rpy::streams::Stream::Interval& interval,
        rpy::resolution_t resolution,
        const rpy::streams::Stream::Context& ctx
) const
{
    auto amended_query = refine_interval(interval);
    if (!amended_query) { return zero_lie(ctx); }

    return log_signature_impl(amended_query->first, resolution, ctx);
}
Stream::FreeTensor Stream::signature(
        const Stream::Interval& interval,
        const Stream::Context& ctx
) const
{
    return ctx.to_signature(log_signature(interval, ctx));
}

rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature(
        const rpy::streams::Stream::Interval& interval,
        rpy::resolution_t resolution,
        const rpy::streams::Stream::Context& ctx
) const
{
    return ctx.to_signature(log_signature(interval, resolution, ctx));
}

Stream::FreeTensor Stream::signature_derivative(
        const Stream::perturbation_list_t& perturbations,
        const Stream::Context& ctx
) const
{
    const auto& md = metadata();
    std::vector<algebra::DerivativeComputeInfo> info;
    info.reserve(perturbations.size());
    for (auto&& pert : perturbations) {
        info.push_back({log_signature(pert.first, ctx), pert.second}
        );
    }
    return ctx.sig_derivative(info, md.cached_vector_type);
}

rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature_derivative(
        const rpy::streams::Stream::Interval& domain,
        const rpy::streams::Stream::Lie& perturbation,
        const rpy::streams::Stream::Context& ctx
) const
{
    const auto& md = metadata();
    algebra::DerivativeComputeInfo info{
            log_signature(domain, ctx),
            perturbation};

    return ctx.sig_derivative({std::move(info)}, md.cached_vector_type);
}

rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature_derivative(
        const rpy::streams::Stream::Interval& domain,
        const rpy::streams::Stream::Lie& perturbation,
        rpy::resolution_t resolution,
        const rpy::streams::Stream::Context& ctx
) const
{
    const auto& md = metadata();
    algebra::DerivativeComputeInfo info{
            log_signature(domain, resolution, ctx),
            perturbation};
    return ctx.sig_derivative({std::move(info)}, md.cached_vector_type);
}

rpy::streams::Stream::FreeTensor rpy::streams::Stream::signature_derivative(
        const rpy::streams::Stream::perturbation_list_t& perturbations,
        rpy::resolution_t resolution,
        const rpy::streams::Stream::Context& ctx
) const
{
    const auto& md = metadata();
    std::vector<algebra::DerivativeComputeInfo> info;
    info.reserve(perturbations.size());
    for (auto&& pert : perturbations) {
        info.push_back({log_signature(pert.first, resolution, ctx), pert.second}
        );
    }
    return ctx.sig_derivative(info, md.cached_vector_type);
}

Stream Stream::simplify(
        const intervals::Partition& partition,
        resolution_t resolution,
        const Stream::Context& ctx
) const
{
    using LiePiece = typename PiecewiseAbelianStream::LiePiece;

    std::vector<LiePiece> pieces;
    const auto partition_size = partition.size();
    pieces.reserve(partition_size);

    for (dimn_t i = 0; i < partition_size; ++i) {
        const auto interval = partition[i];
        pieces.emplace_back(interval, log_signature(interval, resolution, ctx));
    }

    StreamMetadata new_md(metadata());
    new_md.default_resolution = resolution;
    new_md.default_context = &ctx;

    return Stream(PiecewiseAbelianStream(
            std::move(pieces),
            std::move(new_md),
            p_impl->get_schema()
    ));
}

#define RPY_EXPORT_MACRO ROUGHPY_STREAMS_EXPORT
#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::Stream

#include <roughpy/platform/serialization_instantiations.inl>

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

//
// Created by user on 10/03/23.
//

#include <roughpy/streams/piecewise_abelian_stream.h>
#include <roughpy/streams/schema.h>

using namespace rpy;
using namespace rpy::streams;

using rpy::algebra::Context;
using rpy::algebra::Lie;
using rpy::intervals::Interval;
using rpy::intervals::RealInterval;

PiecewiseAbelianStream::PiecewiseAbelianStream(
        std::vector<LiePiece>&& data, StreamMetadata&& md
)
    : StreamInterface(std::move(md)), m_data(std::move(data))
{
    //    // first sort so we know the inf of each interval are in order
    //    auto sort_fun = [](const LiePiece &a, const LiePiece &b) {
    //        return a.first.inf() < b.first.inf();
    //    };
    //    std::sort(data.begin(), data.end(), sort_fun);
    //
    //    m_data.reserve(data.size());
    //    auto next = data.begin();
    //    auto it = next++;
    //    auto end = data.end();
    //    while (next != data.end()) {
    //        auto curr_i = it->first.inf(), curr_s = it->first.sup();
    //        auto next_i = next->first.inf(), next_s = next->first.sup();
    //        if (next_i < curr_s) {
    //            /*
    //         * If the interval of the next piece overlaps with the current
    //         piece interval
    //         * then we need to do a little hacking/slashing to make sure our
    //         data is correct.
    //         */
    //            RealInterval new_curr_interval(curr_i, next_i);
    //            auto new_lie = compute_lie_piece(*it, new_curr_interval);
    //            it->first = new_curr_interval;
    //            it->second.sub_inplace(new_lie);
    //            m_data.emplace_back(std::move(new_curr_interval),
    //            std::move(new_lie));
    //            /*
    //         * At this stage the part [a---[b has been sorted , so we need to
    //         decide what the next
    //         * iteration will look like. This means separating the
    //         intersecting parts of [b---a)
    //         * and on a)---b) (if it exists).
    //         */
    //
    //            if (curr_s < next_s) {
    //                // [a---[b--a)----b)
    //                // it->second is going to be replaced by the part on
    //                [b--a) and
    //                // next->second replace by the part on a)----b)
    //                RealInterval next_curr_interval(next_i, curr_s);
    //                new_lie = compute_lie_piece(*next, next_curr_interval);
    //
    //                it->first = RealInterval(next_i, curr_s);
    //                it->second.add_inplace(new_lie);
    //                next->first = RealInterval(curr_s, next_s);
    //                next->second.sub_inplace(new_lie);
    //
    //                // Increment next so the next iteration checks the
    //                interaction of the new it with next++
    //                // Do not increment it yet.
    //                ++next;
    //                continue;
    //            } else if (curr_s == next_s) {
    //                // [a---[b-------ab)
    //                // it->second is to be discarded, next->second replaced by
    //                part on [b----ab). next->second.add_inplace(it->second);
    //            } else {
    //                // [a---[b---b)---a)
    //                // it->second replaced by part on [b---b)
    //                // next-second replaced by part on b)---a)
    //                new_lie = compute_lie_piece(*it, next->first);
    //                std::swap(*it, *next);
    //
    //                it->second.add_inplace(new_lie);
    //                next->first = RealInterval(next_s, curr_s);
    //                next->second.sub_inplace(new_lie);
    //
    //                // Increment next so the next iteration checks the
    //                interaction of the new it with next++
    //                // Do not increment it yet.
    //                ++next;
    //                continue;
    //            }
    //        } else {
    //            // Here the intervals of definition are disjoint, so we can
    //            just push onto m_data. m_data.push_back(std::move(*it));
    //        }
    //
    //        if (++it == next) {
    //            ++next;
    //        }
    //    }
    //    for (; it != end; ++it) {
    //        m_data.push_back(std::move(*it));
    //    }

    const auto& meta = metadata();
    auto schema = std::make_shared<streams::StreamSchema>();
    auto& info = schema->insert_lie("");
    info.set_lie_info(
            meta.width, meta.default_context->depth(), meta.cached_vector_type
    );
    set_schema(std::move(schema));
}
PiecewiseAbelianStream::PiecewiseAbelianStream(
        std::vector<LiePiece>&& arg, StreamMetadata&& md,
        std::shared_ptr<StreamSchema> schema
)
    : StreamInterface(std::move(md), std::move(schema)), m_data(std::move(arg))
{}

bool PiecewiseAbelianStream::empty(const Interval& interval) const noexcept
{
    return StreamInterface::empty(interval);
}
algebra::Lie PiecewiseAbelianStream::log_signature_impl(
        const Interval& domain, const Context& ctx
) const
{
    std::vector<algebra::Lie> lies;
    lies.reserve(4);

    auto a = domain.inf(), b = domain.sup();
    for (const auto& piece : m_data) {
        // data is in order, so if we are already past the end of the request
        // interval, then we are done so break.
        auto pa = piece.first.inf(), pb = piece.first.sup();
        if (pa >= b) {
            // [-----) [p----p)
            break;
        }
        if (pb <= a) {
            // [p----p) [-----)
        } else if (pa >= a && pb <= b) {
            // [-----[p-----p)---)
            lies.push_back(piece.second);
        } else if (pb <= pb) {
            // [p---[---p)-----)
            lies.push_back(piece.second.smul(to_multiplier_upper(piece.first, a)
            ));
        } else if (pa >= a && pb > b) {
            // [---[p----)----p)
            lies.push_back(piece.second.smul(to_multiplier_lower(piece.first, b)
            ));
        }
    }

    const auto& md = metadata();
    return ctx.cbh(lies, md.cached_vector_type);
}

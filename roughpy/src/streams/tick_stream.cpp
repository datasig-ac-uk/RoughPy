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

#include "tick_stream.h"

#include <map>
#include <set>

#include <roughpy/algebra/lie.h>
#include <roughpy/intervals/interval.h>
#include <roughpy/streams/stream.h>
#include <roughpy/streams/stream_construction_helper.h>
#include <roughpy/streams/tick_stream.h>

#include "args/convert_timestamp.h"
#include "args/kwargs_to_path_metadata.h"
#include "args/parse_schema.h"
#include "r_py_tick_construction_helper.h"
#include "scalars/scalar_type.h"
#include "scalars/scalars.h"
#include "streams/stream.h"

using namespace rpy;
using namespace pybind11::literals;
using namespace std::literals::string_literals;

using helper_t = streams::StreamConstructionHelper;

static const char* TICK_STREAM_DOC = R"rpydoc(Tick stream
)rpydoc";

// static void parse_data_to_ticks(helper_t &helper,
//                                 const py::handle &data,
//                                 const py::kwargs &kwargs);

static py::object construct(const py::object& data, const py::kwargs& kwargs)
{

    auto pmd = python::kwargs_to_metadata(kwargs);
    auto& schema = pmd.schema;

    if (!schema) { schema = std::make_shared<streams::StreamSchema>(); }

    py::object parser;
    if (kwargs.contains("parser")) {
        parser = kwargs["parser"](schema);
    } else {
        auto tick_helpers_mod
                = py::module_::import("roughpy.streams.tick_stream");
        parser = tick_helpers_mod.attr("StandardTickDataParser")(schema);
    }

    parser.attr("parse_data")(data);

    auto& helper
            = parser.attr("helper").cast<python::RPyTickConstructionHelper&>();

    const auto& ticks = helper.ticks();
    if (ticks.empty()) {
        RPY_THROW(py::value_error, "tick data cannot be empty");
    }

    //    if (!schema->is_final()) {
    //        python::parse_data_into_schema(schema, data);
    //        pmd.width = schema->width();
    //        schema->finalize();
    //    }
    if (!pmd.ctx) {
        pmd.width = schema->width();
        if (pmd.width == 0 || pmd.depth == 0 || pmd.scalar_type == nullptr) {
            RPY_THROW(
                    py::value_error,
                    "either ctx or width, depth, and dtype must be provided"
            );
        }
        pmd.ctx = algebra::get_context(pmd.width, pmd.depth, pmd.scalar_type);
    } else if (pmd.width != pmd.ctx->width()) {
        // Recalibrate the width to match the data
        pmd.ctx = pmd.ctx->get_alike(pmd.width, pmd.depth, pmd.scalar_type);
    }

    //    helper_t helper(
    //        pmd.ctx,
    //        schema,
    //        static_cast<bool>(pmd.vector_type)
    //            ? *pmd.vector_type
    //            : algebra::VectorType::Sparse);

    //    parse_data_to_ticks(helper, data, kwargs);

    //    python::PyToBufferOptions options;
    //    options.type = pmd.scalar_type;
    //    options.max_nested = 2;
    //    options.allow_scalar = false;
    //
    //    auto buffer = python::py_to_buffer(data, options);

    //    scalars::ScalarStream stream(options.type);
    //    stream.reserve_size(options.shape.size());

    //    auto result = streams::Stream(
    //        streams::TickStream(
    //            std::move(helper),
    //            {pmd.width,
    //             pmd.support ? *pmd.support : intervals::RealInterval(0, 1),
    //             pmd.ctx,
    //             pmd.scalar_type,
    //             pmd.vector_type ? *pmd.vector_type :
    //             algebra::VectorType::Dense, pmd.resolution, pmd.interval_type
    //            },
    //            pmd.resolution));

    streams::StreamMetadata meta{
            pmd.width,
            pmd.support ? *pmd.support : intervals::RealInterval(0, 1),
            pmd.ctx,
            pmd.scalar_type,
            pmd.vector_type ? *pmd.vector_type : algebra::VectorType::Dense,
            pmd.resolution,
            pmd.interval_type};

    std::set<param_t> index;
    std::map<intervals::DyadicInterval, algebra::Lie> raw_data;

    // For value types, we need to keep track of the last value that appeared
    std::vector<algebra::Lie> previous_values(
            pmd.width, pmd.ctx->zero_lie(meta.cached_vector_type)
    );
    for (const auto& tick : ticks) {
        const intervals::DyadicInterval di(
                tick.timestamp, pmd.resolution, pmd.interval_type
        );

        auto lie_elt = pmd.ctx->zero_lie(meta.cached_vector_type);

        auto channel_it = schema->find(tick.label);
        RPY_DBG_ASSERT(channel_it != schema->end());
        auto& channel = channel_it->second;
        auto idx = schema->label_to_stream_dim(tick.label);
        auto key = schema->label_to_lie_key(tick.label);
        switch (tick.type) {
            case streams::ChannelType::Increment:
                lie_elt[key]
                        += python::py_to_scalar(pmd.scalar_type, tick.data);
                break;
            case streams::ChannelType::Value:
                if (channel.is_lead_lag()) {
                    auto lag_key = key + 1;
                    const auto& prev_lead = previous_values[idx];
                    const auto& prev_lag = previous_values[idx + 1];

                    auto lead = pmd.ctx->zero_lie(meta.cached_vector_type);
                    auto lag = pmd.ctx->zero_lie(meta.cached_vector_type);

                    lead[key]
                            += python::py_to_scalar(pmd.scalar_type, tick.data);
                    lead[lag_key]
                            += python::py_to_scalar(pmd.scalar_type, tick.data);

                    lie_elt = pmd.ctx->cbh(
                            lead.sub(prev_lead), lag.sub(prev_lag),
                            meta.cached_vector_type
                    );

                    previous_values[idx] = std::move(lead);
                    previous_values[idx + 1] = std::move(lag);

                    break;
                } else {
                    const auto& prev_val = previous_values[idx];
                    auto new_val = pmd.ctx->zero_lie(meta.cached_vector_type);

                    new_val[key]
                            += python::py_to_scalar(pmd.scalar_type, tick.data);

                    lie_elt = new_val.sub(prev_val);
                    previous_values[idx] = std::move(new_val);

                    break;
                }
            case streams::ChannelType::Categorical: {
                lie_elt[key] += pmd.scalar_type->one();
                break;
            }
            case streams::ChannelType::Lie:
                RPY_THROW(
                        py::value_error, "Lie tick types currently not allowed"
                );
        }

        auto& existing = raw_data[di];
        if (existing) {
            existing = pmd.ctx->cbh(existing, lie_elt, meta.cached_vector_type);
        } else {
            existing = std::move(lie_elt);
        }

        index.insert(di.included_end());
    }

    streams::Stream result(streams::TickStream(
            std::vector<param_t>(index.begin(), index.end()),
            std::move(raw_data), pmd.resolution, pmd.schema, std::move(meta)
    ));

    return py::reinterpret_steal<py::object>(
            python::RPyStream_FromStream(std::move(result))
    );
}

void python::init_tick_stream(py::module_& m)
{

    init_tick_construction_helper(m);

    py::class_<streams::TickStream> klass(m, "TickStream", TICK_STREAM_DOC);

    klass.def_static("from_data", &construct, "data"_a);
}

/*
 * Tick data comes as a sequence of values of type:
 *   TickData := Tuple[TimeStamp, TickItem]                     (time, data)
 *             | Tuple[TimeStamp, String, TickItem]             (time, label,
 * data) | Tuple[TimeStamp, String, DataType, TickValue]  (time, label, type,
 * data)
 *
 *   TickItem := TickValue
 *             | Tuple[String, TickValue]
 *             | List[TickValue]
 *             | List[Tuple[String, TickValue]]
 *             | Dict[String, TickValue]
 *
 *   TickValue := Value | List[Value]
 *
 */

namespace {

using streams::ChannelType;
//
// inline void insert_increment(helper_t &helper,
//                             param_t timestamp,
//                             string_view label,
//                             const py::object &value) {
//    scalars::Scalar val(helper.ctype(), 0);
//    python::assign_py_object_to_scalar(val.to_mut_pointer(), value);
//    helper.add_increment(timestamp, label, std::move(val));
//}
//
// inline void insert_value(helper_t &helper,
//                         param_t timestamp,
//                         string_view label,
//                         const py::object &value) {
//    scalars::Scalar val(helper.ctype());
//    python::assign_py_object_to_scalar(val.to_mut_pointer(), value);
//    helper.add_value(timestamp, label, std::move(val));
//}
//
// inline void insert_categorical(helper_t &helper,
//                               param_t timestamp,
//                               string_view label,
//                               const py::object &value) {
//    helper.add_categorical(timestamp, label, value.cast<string>());
//}
//
// inline void insert_lie(helper_t &helper,
//                       param_t timestamp,
//                       string_view label,
//                       const py::object &value) {
//    helper.add_categorical(timestamp, label, value.cast<string>());
//}
//
// inline void handle_tick_value(helper_t &helper,
//                              param_t timestamp,
//                              string_view label,
//                              ChannelType type,
//                              const py::object &tick_value) {
//    switch (type) {
//        case ChannelType::Increment:
//            insert_increment(helper, timestamp, label, tick_value);
//            break;
//        case ChannelType::Value:
//            insert_value(helper, timestamp, label, tick_value);
//            break;
//        case ChannelType::Categorical:
//            insert_categorical(helper, timestamp, label, tick_value);
//            break;
//        case ChannelType::Lie:
//            insert_lie(helper, timestamp, label, tick_value);
//            break;
//    }
//}
//
// inline void handle_labeled_data(helper_t &helper,
//                                param_t timestamp,
//                                string_view label,
//                                const py::object &tick_value) {
//    auto type = helper.type_of(label);
//    if (!type) {
//        RPY_THROW(py::value_error,"unexpected label " + string(label) + " in
//        tick data");
//    }
//
//    handle_tick_value(helper, timestamp, label, *type, tick_value);
//}
//
// void handle_timestamp_pair(helper_t &helper, param_t timestamp, py::object
// tick_item) {
//    /*
//     * The tick object must be either:
//     *      - a label, value pair,
//     *      - a label, type, value triple,
//     *      - a label_value_list,
//     *      - a type, label_value_list pair,
//     *      - a dictionary of [label, value] pairs
//     */
//    if (py::isinstance<py::dict>(tick_item)) {
//        for (auto &&[label, tick_value] :
//        py::reinterpret_steal<py::dict>(tick_item)) {
//            auto value = py::reinterpret_borrow<py::object>(tick_value);
//            handle_labeled_data(helper,
//                                timestamp,
//                                label.cast<string_view>(),
//                                value);
//        }
//    } else if (py::isinstance<py::tuple>(tick_item)) {
//        auto tuple_item = py::reinterpret_steal<py::tuple>(tick_item);
//        auto len = py::len(tuple_item);
//        auto label = tuple_item[0].cast<string>();
//
//        py::object value;
//        if (len == 2) {
//            value = py::reinterpret_borrow<py::object>(tuple_item[1]);
//        } else if (len == 3) {
//            value = py::reinterpret_borrow<py::object>(tuple_item[2]);
//        } else {
//            RPY_THROW(py::value_error,"expected tuple (label, data) or (label,
//            type, data)");
//        }
//
//        if (py::isinstance<py::sequence>(value)) {
//            for (auto&& tick_value :
//            py::reinterpret_steal<py::sequence>(value)) {
//                auto inner = py::reinterpret_borrow<py::object>(tick_value);
//                handle_timestamp_pair(helper, timestamp, inner);
//            }
//        } else {
//            handle_labeled_data(helper, timestamp, label, value);
//        }
//
//    }
//}
//
// inline void handle_dict_tick_stream(helper_t &helper, const py::dict &ticks)
// {
//    for (auto [it_time, it_value] : ticks) {
//        auto timestamp = python::convert_timestamp(
//            py::reinterpret_borrow<py::object>(it_time));
//
//        auto value = py::reinterpret_borrow<py::object>(it_value);
//        handle_timestamp_pair(helper,
//                              timestamp,
//                              std::move(value));
//    }
//}
//
// inline void handle_tuple_sequence_tick_stream(helper_t &helper, const
// py::sequence &ticks) {
//    for (auto &&it_value : ticks) {
//        // Use sequence instead of tuple for inner items, to allow lists
//        instead RPY_CHECK(py::isinstance<py::sequence>(it_value)); auto inner
//        = py::reinterpret_borrow<py::sequence>(it_value); auto len =
//        py::len(inner);
//
//        RPY_CHECK(len > 1 && len <= 4);
//        auto timestamp = python::convert_timestamp(
//            py::reinterpret_borrow<py::object>(inner[0]));
//        auto right = inner[py::slice(1, {}, {})];
//        handle_timestamp_pair(helper, timestamp, right);
//    }
//}

}// namespace

// void parse_data_to_ticks(helper_t &helper, const py::handle &data, const
// py::kwargs &kwargs) {
//
//     if (py::isinstance<py::dict>(data)) {
//         handle_dict_tick_stream(helper,
//         py::reinterpret_borrow<py::dict>(data));
//     } else if (py::isinstance<py::sequence>(data)) {
//         handle_tuple_sequence_tick_stream(
//             helper,
//             py::reinterpret_borrow<py::sequence>(data)
//             );
//     } else {
//         RPY_THROW(py::type_error, "expected dict or sequence of pairs");
//     }
//
// }

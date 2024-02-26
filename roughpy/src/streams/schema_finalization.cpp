// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 22/08/23.
//

#include "schema_finalization.h"

using namespace rpy;
using namespace rpy::python;

/**
 * @brief Perform the last minute setup of the schema and finalize
 *
 * Last minute setup of the schema:
 *  - If needed, set the include time as channel flag.
 *
 *  (Space for more things in the future.)
 *
 * @param schema Schema to set up and finalize
 * @param pmd Path metadata parsed from arguments
 */
static inline void
last_minute_setup(streams::StreamSchema& schema, PyStreamMetaData& pmd)
{

    if (schema.is_final()) { return; }

    // Now is the time to make sure the schema width is correct
    RPY_DBG_ASSERT(pmd.width != 0);
    for (auto i = static_cast<deg_t>(schema.width()); i < pmd.width; ++i) {
        switch (pmd.default_channel_type) {
            case rpy::streams::ChannelType::Increment:
                schema.insert_increment("");
                break;
            case rpy::streams::ChannelType::Value:
                schema.insert_value("");
                break;
            case rpy::streams::ChannelType::Categorical:
                schema.insert_categorical("");
                break;
            case rpy::streams::ChannelType::Lie: schema.insert_lie(""); break;
        }
    }

    if (pmd.include_param_as_data) {
        schema.parametrization()->add_as_channel();
    }

    schema.finalize(0);
}

void python::finalize_schema(PyStreamMetaData& pmd)
{
    auto& schema = *pmd.schema;

    // does nothing it the schema is already final.
    last_minute_setup(schema, pmd);

    // At this point the schema must be final.
    RPY_DBG_ASSERT(schema.is_final());

    auto schema_width = static_cast<deg_t>(schema.width());

    if (schema_width == 0) {
        RPY_THROW(py::value_error, "the stream has no data channels");
    }

    if (pmd.include_param_as_data && schema_width == 1) {
        RPY_THROW(
                py::value_error,
                "time cannot be the only data channel "
                "in a stream"
        );
    }

    if (pmd.width == 0) {
        // No width was specified, set according to schema
        pmd.width = schema_width;
    } else if (pmd.width < schema_width) {
        // It's fine if the provided width is the number of incoming data
        // channels without time, provided that time is the only difference
        if (!pmd.include_param_as_data || pmd.width + 1 < schema_width) {
            RPY_THROW(
                    py::value_error,
                    "specified width is smaller than the "
                    "number of number of channels of the data"
            );
        }

        // update pmd.width for checks below and any other checks the
        // constructor might use
        pmd.width = schema_width;
    }

    /*
     * At this point, the value of pmd.width cannot be zero and cannot be 1,
     * and time cannot be the only data channel. Now what we need to do is
     * check if a context has been set, and set if not, and perform any
     * additional checks that are required for compatibility.
     *
     * The first thing to do, if it is not already set, is to figure out what
     * scalar type is most appropriate
     */
    if (pmd.scalar_type == nullptr) {
        pmd.scalar_type = schema.get_most_appropriate_scalar_type();
    }

    // An exception should have been thrown if no suitable scalar set
    RPY_DBG_ASSERT(pmd.scalar_type != nullptr);

    // If the depth is not set, then use the default
    if (pmd.depth == 0) {
        // TODO: reach out to config to get the default depth
        pmd.depth = 2;
    }

    RPY_DBG_ASSERT(pmd.depth > 0);

    if (!pmd.ctx) {
        // If there is no provided context, set it.
        pmd.ctx = algebra::get_context(pmd.width, pmd.depth, pmd.scalar_type);
    } else if (pmd.width != pmd.ctx->width()
               || pmd.depth != pmd.ctx->depth()
               || pmd.scalar_type != pmd.ctx->ctype()) {
        // Recalibrate to match the data
        pmd.ctx = pmd.ctx->get_alike(pmd.width, pmd.depth, pmd.scalar_type);
    }


}

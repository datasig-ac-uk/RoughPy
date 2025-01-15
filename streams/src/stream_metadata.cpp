#include "stream_metadata.h"


#include "roughpy/core/macros.h"
#include "roughpy/core/check.h"

using namespace rpy;
using namespace rpy::streams;


StreamMetadataBuilder::StreamMetadataBuilder(
    const StreamMetadata* existing)
{
    if (!existing) { p_metadata = std::make_shared<StreamMetadata>(); } else {
        p_metadata = std::make_shared<StreamMetadata>(*existing);
        auto ctx = existing->p_default_context;
        RPY_CHECK(ctx);

        m_default_depth = ctx->depth();
        p_ctype = ctx->ctype();
    }
}

StreamMetadataBuilder& StreamMetadataBuilder::
add_channel(string channel_name)
{
    p_metadata->m_channel_names.push_back(std::move(channel_name));
    return *this;
}

std::shared_ptr<StreamMetadata> StreamMetadataBuilder::build()
{
    RPY_CHECK(!p_metadata->m_channel_names.empty());

    const auto stream_dimension = static_cast<deg_t>(p_metadata->m_channel_names
        .size());

    if (!p_ctype) { p_ctype = *scalars::ScalarType::of<double>(); }

    p_metadata->p_default_context = algebra::get_context(
        stream_dimension,
        m_default_depth,
        p_ctype);

    return std::move(p_metadata);
}
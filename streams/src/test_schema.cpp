// Copyright (c) 2023 Datasig Developers. All rights reserved.
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
// Created by user on 25/05/23.
//

#include <gtest/gtest.h>
#include <roughpy/streams/schema.h>

#include <sstream>

using namespace rpy;
using namespace rpy::streams;

TEST(Schema, TestLabelCompareEqual)
{
    string first("first");

    EXPECT_TRUE(StreamSchema::compare_labels(first, first));
}

TEST(Schema, TestLabelCompareDifferentLengths)
{
    string first("first");
    string first1("first:1");

    // this is true because first1 = first + ":" + ...
    EXPECT_TRUE(StreamSchema::compare_labels(first, first1));
    // This fails because ref_label.size() < item_label.size()
    EXPECT_FALSE(StreamSchema::compare_labels(first1, first));
}

TEST(Schema, TestLabelCompareNonTerminatedCorrectly)
{
    string first("first");
    string first1("first1");

    EXPECT_FALSE(StreamSchema::compare_labels(first, first1));
}

TEST(Schema, TestLabelCompareDifferentStrings)
{
    string first("first");
    string firnt("firnt");

    EXPECT_FALSE(StreamSchema::compare_labels(first, firnt));
    EXPECT_FALSE(StreamSchema::compare_labels(firnt, first));
}

TEST(Schema, TestLabelCompareEmptyRefString)
{
    EXPECT_FALSE(StreamSchema::compare_labels("", ":test"));
}

TEST(Schema, TestStreamChannelIncrementSerialization)
{
    StreamChannel channel(ChannelType::Increment);

    std::stringstream ss;
    {
        archives::JSONOutputArchive oarch(ss);
        oarch(channel);
    }

    StreamChannel in_channel;
    {
        archives::JSONInputArchive iarch(ss);
        iarch(in_channel);
    }

    EXPECT_EQ(in_channel.type(), channel.type());
}

TEST(Schema, TestStreamChannelValueSerialization)
{
    StreamChannel channel(ChannelType::Value);

    std::stringstream ss;
    {
        archives::JSONOutputArchive oarch(ss);
        oarch(channel);
    }

    StreamChannel in_channel;
    {
        archives::JSONInputArchive iarch(ss);
        iarch(in_channel);
    }

    EXPECT_EQ(in_channel.type(), channel.type());
}

TEST(Schema, TestStreamChannelCategoricalSerialization)
{
    StreamChannel channel(ChannelType::Categorical);
    channel.add_variant("first").add_variant("second");

    std::stringstream ss;
    {
        archives::JSONOutputArchive oarch(ss);
        oarch(channel);
    }

    StreamChannel in_channel;
    {
        archives::JSONInputArchive iarch(ss);
        iarch(in_channel);
    }

    EXPECT_EQ(in_channel.type(), channel.type());
    EXPECT_EQ(in_channel.get_variants(), channel.get_variants());
}

namespace {

class SchemaTests : public ::testing::Test
{
protected:
    StreamSchema schema;

    SchemaTests()
    {
        schema.reserve(3);
        schema.insert_categorical("a")
                .add_variant("first")
                .add_variant("second")
                .add_variant("third");
        schema.insert_value("c").set_lead_lag(true);
        schema.insert_increment("b");
    }
};

}// namespace

TEST_F(SchemaTests, TestSchemaFind)
{
    const auto end = schema.end();
    auto finda = schema.find("a");
    ASSERT_NE(finda, end);
    EXPECT_EQ(finda->first, "a");

    auto finda_first = schema.find("a:first");
    ASSERT_NE(finda_first, end);
    EXPECT_EQ(finda_first->first, "a");

    auto findc = schema.find("c:lead");
    ASSERT_NE(findc, end);
    EXPECT_EQ(findc->first, "c");

    auto findb = schema.find("b");
    ASSERT_NE(findb, end);
    EXPECT_EQ(findb->first, "b");
}

TEST_F(SchemaTests, TestSchemaWidthCalculation)
{
    EXPECT_EQ(schema.width(), 3 + 2 + 1);
}

TEST_F(SchemaTests, TestChannelToStreamDim)
{
    EXPECT_EQ(schema.channel_to_stream_dim(0), 0);
    EXPECT_EQ(schema.channel_to_stream_dim(1), 3);
    EXPECT_EQ(schema.channel_to_stream_dim(2), 5);
}

TEST_F(SchemaTests, TestChannelVariantToStreamDim)
{
    EXPECT_EQ(schema.channel_variant_to_stream_dim(0, 0), 0);
    EXPECT_EQ(schema.channel_variant_to_stream_dim(0, 1), 1);
    EXPECT_EQ(schema.channel_variant_to_stream_dim(0, 2), 2);
    EXPECT_EQ(schema.channel_variant_to_stream_dim(1, 0), 3);
    EXPECT_EQ(schema.channel_variant_to_stream_dim(1, 1), 4);
    EXPECT_EQ(schema.channel_variant_to_stream_dim(2, 0), 5);
}

TEST_F(SchemaTests, TestStreamDimToChannel)
{
    using pair_t = pair<dimn_t, dimn_t>;
    EXPECT_EQ(schema.stream_dim_to_channel(0), pair_t(0, 0));
    EXPECT_EQ(schema.stream_dim_to_channel(1), pair_t(0, 1));
    EXPECT_EQ(schema.stream_dim_to_channel(2), pair_t(0, 2));
    EXPECT_EQ(schema.stream_dim_to_channel(3), pair_t(1, 0));
    EXPECT_EQ(schema.stream_dim_to_channel(4), pair_t(1, 1));
    EXPECT_EQ(schema.stream_dim_to_channel(5), pair_t(2, 0));
}

TEST_F(SchemaTests, TestLabelToStreamDimIncrement)
{
    EXPECT_EQ(schema.label_to_stream_dim("b"), 5);
}

TEST_F(SchemaTests, TestLabelToStreamDimValue)
{
    EXPECT_EQ(schema.label_to_stream_dim("c"), 3);
    EXPECT_EQ(schema.label_to_stream_dim("c:lead"), 3);
    EXPECT_EQ(schema.label_to_stream_dim("c:lag"), 4);
}

TEST_F(SchemaTests, TestLabelToStreamDimCategorical)
{
    EXPECT_EQ(schema.label_to_stream_dim("a"), 0);
    EXPECT_EQ(schema.label_to_stream_dim("a:first"), 0);
    EXPECT_EQ(schema.label_to_stream_dim("a:second"), 1);
    EXPECT_EQ(schema.label_to_stream_dim("a:third"), 2);
}

TEST_F(SchemaTests, TestLabelOfStreamDim)
{
    EXPECT_EQ(schema.label_of_stream_dim(0), "a:first");
    EXPECT_EQ(schema.label_of_stream_dim(1), "a:second");
    EXPECT_EQ(schema.label_of_stream_dim(2), "a:third");
    EXPECT_EQ(schema.label_of_stream_dim(3), "c:lead");
    EXPECT_EQ(schema.label_of_stream_dim(4), "c:lag");
    EXPECT_EQ(schema.label_of_stream_dim(5), "b");
}

TEST_F(SchemaTests, TestLabelOfChannelId)
{
    EXPECT_EQ(schema.label_of_channel_id(0), "a");
    EXPECT_EQ(schema.label_of_channel_id(1), "c");
    EXPECT_EQ(schema.label_of_channel_id(2), "b");
}

TEST_F(SchemaTests, TestLabelOfChannelVariant)
{
    EXPECT_EQ(schema.label_of_channel_variant(0, 0), "a:first");
    EXPECT_EQ(schema.label_of_channel_variant(0, 1), "a:second");
    EXPECT_EQ(schema.label_of_channel_variant(0, 2), "a:third");
    EXPECT_EQ(schema.label_of_channel_variant(1, 0), "c:lead");
    EXPECT_EQ(schema.label_of_channel_variant(1, 1), "c:lag");
    EXPECT_EQ(schema.label_of_channel_variant(2, 0), "b");
}

TEST_F(SchemaTests, TestSerializationOfSchema)
{
    std::stringstream ss;
    {
        archives::JSONOutputArchive oarch(ss);
        oarch(schema);
    }

    StreamSchema in_schema;
    {
        archives::JSONInputArchive iarch(ss);
        iarch(in_schema);
    }

    EXPECT_EQ(in_schema.size(), schema.size());
    EXPECT_EQ(in_schema.width(), schema.width());
}

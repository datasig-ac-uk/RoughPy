// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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
// Created by user on 26/06/23.
//

#include "standard_random_generator.h"
#include <gtest/gtest.h>
#include <sstream>
#include <string>

using generator_t = rpy::scalars::StandardRandomGenerator<double, pcg64>;
using namespace std::literals::string_literals;

struct my_pcg64 : pcg64 {
    using pcg64::state_;
};

TEST(PCGGenTests, LoadFromTerminatedSStream)
{

    std::stringstream ss("5");
    ss.flags(std::ios_base::dec);
    auto in = ss.get();
    EXPECT_EQ(in, '5');
    auto in2 = ss.get();
    EXPECT_EQ(in2, std::stringstream::traits_type::eof());
}

TEST(PCGenTests, LoadRandomState)
{
    std::stringstream ss(
            "47026247687942121848144207491837523525 117397592171526113268558934119004209487 120436820235895678955951683610125339985\n"s);
    my_pcg64 gen;
    ss >> gen;

    EXPECT_EQ(gen.state_,
              PCG_128BIT_CONSTANT(6528893107350212886ULL,
                                  10831353900920016209ULL));
}

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
// Created by user on 17/04/23.
//

#include "roughpy/platform/configuration.h" // IWYU pragma: associated

#include "roughpy/core/slice.h"  // for Slice
#include "roughpy/core/types.h"  // for string_view

using namespace rpy;

class Configuration::State
{
};

rpy::Configuration::Configuration() : p_state(new State) {}
rpy::Configuration::~Configuration() = default;

string_view Configuration::get_raw_config_value(string_view property) const
{
    return string_view();
}

const Configuration& rpy::get_config()
{
    static const Configuration config;
    return config;
}

fs::path rpy::Configuration::stream_cache_dir() const {
    auto path = fs::current_path() / ".stream_cache";
    return path;
}

Slice<const fs::path> rpy::Configuration::kernel_source_search_dirs() const
{
    return {};
}

const fs::path& rpy::Configuration::get_builtin_kernel_dir() const
{
    static fs::path p;
    return p;
}

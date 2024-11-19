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

#ifndef ROUGHPY_PLATFORM_CONFIGURATION_H
#define ROUGHPY_PLATFORM_CONFIGURATION_H

#include <memory>

#include "roughpy/core/macros.h"
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>
#include <roughpy/core/slice.h>


#include "roughpy/platform/roughpy_platform_export.h"

#include "filesystem.h" // IWYU pragma: keep

namespace rpy {


// Forward declaration
class Configuration;

/**
 * @brief Get a reference to the current configuration.
 * @return global configuration reference
 */
ROUGHPY_PLATFORM_EXPORT const Configuration& get_config();

/**
 * @brief Interface for getting RoughPy configuration settings.
 */
class ROUGHPY_PLATFORM_EXPORT Configuration
{

    class State;

    std::unique_ptr<State> p_state;

    Configuration();
    ~Configuration();

    friend ROUGHPY_PLATFORM_EXPORT const Configuration& get_config();

public:
    RPY_NO_DISCARD
    string_view get_raw_config_value(string_view property) const;

    template <typename T>
    RPY_NO_DISCARD enable_if_t<is_constructible_v<T, string_view>, T>
    get_config_value(string_view property) const;

    fs::path stream_cache_dir() const;

    // TODO: In the future, this will include methods for finding runtime
    // libraries like libcudart

    Slice<const fs::path> kernel_source_search_dirs() const;
    const fs::path& get_builtin_kernel_dir() const;
};

template <typename T>
RPY_NO_DISCARD enable_if_t<is_constructible_v<T, string_view>, T>
Configuration::get_config_value(string_view property) const
{
    return T(get_raw_config_value(property));
}

}// namespace rpy

#endif// ROUGHPY_PLATFORM_CONFIGURATION_H

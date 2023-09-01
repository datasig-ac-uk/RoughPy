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

#ifndef ROUGHPY_PLATFORM_RUNTIME_LIBRARY_H_
#define ROUGHPY_PLATFORM_RUNTIME_LIBRARY_H_

#include <roughpy/core/macros.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include "filesystem.h"

#if defined(RPY_HAS_STD_FILESYSTEM) && RPY_HAS_STD_FILESYSTEM
#define BOOST_DLL_USE_STD_FS
#endif
#include <boost/dll/shared_library.hpp>


namespace rpy {
namespace platform {

class RuntimeLibrary : private boost::dll::shared_library
{
    using base_t = boost::dll::shared_library;

public:
    using load_mode = boost::dll::load_mode::type;

    explicit RuntimeLibrary(
            const fs::path& path, load_mode mode = load_mode::default_mode
    )
        : base_t(path, mode)
    {}

    RuntimeLibrary(const RuntimeLibrary&) = delete;
    RuntimeLibrary(RuntimeLibrary&&) noexcept = delete;

    virtual ~RuntimeLibrary();

    RuntimeLibrary& operator=(const RuntimeLibrary&) = delete;
    RuntimeLibrary& operator=(RuntimeLibrary&&) noexcept = delete;

    using base_t::location;

protected:
    using base_t::get;
    using base_t::has;
    using base_t::is_loaded;
    using base_t::get_alias;
};

namespace dtl {
RPY_EXPORT RuntimeLibrary* get_runtime_library(const fs::path& path);
RPY_EXPORT RuntimeLibrary*
register_runtime_library(std::unique_ptr<RuntimeLibrary>&& lib);
}// namespace dtl

template <typename Class>
enable_if_t<is_base_of<Class, RuntimeLibrary>::value, Class*>
load_runtime_library(const fs::path& path)
{
    auto* lib = dtl::get_runtime_library(path);
    if (lib == nullptr) {
        lib = dtl::register_runtime_library(std::make_unique<Class>(path));
    }
    auto* lib_as_class = dynamic_cast<Class*>(lib);
    if (lib_as_class == nullptr) {
        RPY_THROW(std::runtime_error, "could not load " + string(path));
    }
    return lib_as_class;
}

}// namespace platform
}// namespace rpy

#endif// ROUGHPY_PLATFORM_RUNTIME_LIBRARY_H_

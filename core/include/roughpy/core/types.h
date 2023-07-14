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

#ifndef ROUGHPY_CORE_IMPLEMENTATION_TYPES_H_
#define ROUGHPY_CORE_IMPLEMENTATION_TYPES_H_

#include "macros.h"

#include <cstdint>
#include <string>
#include <utility>

#ifdef RPY_CPP_17
#  include <optional>
#  include <string_view>
#else
#  include <boost/optional.hpp>
#  include <boost/utility/string_view.hpp>
#endif

namespace rpy {

using std::int8_t;
using std::uint8_t;
using byte = uint8_t;
using std::int16_t;
using std::int32_t;
using std::int64_t;
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;

using let_t = uint16_t;
using dimn_t = std::size_t;
using idimn_t = std::ptrdiff_t;
using deg_t = int;
using key_type = std::size_t;
using param_t = double;
using scalar_t = double;

using resolution_t = int;
using dyadic_multiplier_t = int;
using dyadic_depth_t = resolution_t;

using accuracy_t = param_t;

using std::pair;
using std::string;

#ifdef RPY_CPP_17
using std::optional;
using std::string_view;
#else
using boost::optional;
using boost::string_view;
#endif

}// namespace rpy

#endif// ROUGHPY_CORE_IMPLEMENTATION_TYPES_H_

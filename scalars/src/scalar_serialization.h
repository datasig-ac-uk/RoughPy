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

#ifndef ROUGHPY_SCALARS_SCALAR_SERIALIZATION_H_
#define ROUGHPY_SCALARS_SCALAR_SERIALIZATION_H_

#include <roughpy/core/helpers.h>
// #include <roughpy/platform/serialization.h>
//
// #include "scalar_implementations/arbitrary_precision_rational.h"
// #include "scalar_implementations/bfloat.h"
// #include "scalar_implementations/complex.h"
// #include "scalar_implementations/half.h"
// #include "scalar_implementations/poly_rational.h"
// #include "scalar_implementations/rational.h"
//
// #include <cereal/types/utility.hpp>
//
// RPY_SERIAL_EXT_LIB_LOAD_FN(::rpy::scalars::Half)
// {
//     using namespace ::rpy;
//     using namespace ::rpy::scalars;
//
//     uint16_t tmp;
//     RPY_SERIAL_SERIALIZE_NVP("value", tmp);
//     value = bit_cast<Half>(tmp);
// }
//
// RPY_SERIAL_EXT_LIB_SAVE_FN(::rpy::scalars::Half)
// {
//     using namespace ::rpy;
//     using namespace ::rpy::scalars;
//     RPY_SERIAL_SERIALIZE_NVP("value", bit_cast<uint16_t>(value));
// }
//
// RPY_SERIAL_EXT_LIB_LOAD_FN(::rpy::scalars::BFloat16)
// {
//     using namespace ::rpy;
//     using namespace ::rpy::scalars;
//
//     uint16_t tmp;
//     RPY_SERIAL_SERIALIZE_NVP("value", tmp);
//     value = bit_cast<BFloat16>(tmp);
// }
//
// RPY_SERIAL_EXT_LIB_SAVE_FN(::rpy::scalars::BFloat16)
// {
//     using namespace ::rpy;
//     using namespace ::rpy::scalars;
//     RPY_SERIAL_SERIALIZE_NVP("value", bit_cast<uint16_t>(value));
// }
//
// RPY_SERIAL_EXT_LIB_LOAD_FN(rpy::scalars::indeterminate_type)
// {
//     using namespace ::rpy::scalars;
//     using packed_type = typename indeterminate_type::packed_type;
//     using integral_type = typename indeterminate_type::integral_type;
//
//     packed_type symbol;
//     RPY_SERIAL_SERIALIZE_VAL(symbol);
//
//     integral_type index;
//     RPY_SERIAL_SERIALIZE_VAL(index);
//
//     value = indeterminate_type(symbol, index);
// }
//
// RPY_SERIAL_EXT_LIB_SAVE_FN(rpy::scalars::indeterminate_type)
// {
//     using namespace ::rpy::scalars;
//     using packed_type = typename indeterminate_type::packed_type;
//     using integral_type = typename indeterminate_type::integral_type;
//     RPY_SERIAL_SERIALIZE_NVP("symbol", static_cast<packed_type>(value));
//     RPY_SERIAL_SERIALIZE_NVP("index", static_cast<integral_type>(value));
// }

#endif// ROUGHPY_SCALARS_SCALAR_SERIALIZATION_H_

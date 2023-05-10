// Copyright (c) 2023 RoughPy Developers. All rights reserved.
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

#ifndef ROUGHPY_ALGEBRA_CONTEXT_FWD_H_
#define ROUGHPY_ALGEBRA_CONTEXT_FWD_H_

#include "algebra_fwd.h"
#include <stdexcept>

#include <roughpy/core/slice.h>
#include <boost/smart_ptr/intrusive_ptr.hpp>

#define RPY_MAKE_VTYPE_SWITCH(VTYPE)                            \
    switch (VTYPE) {                                            \
        case VectorType::Dense:                                 \
            return RPY_SWITCH_FN(VectorType::Dense);            \
        case VectorType::Sparse:                                \
            return RPY_SWITCH_FN(VectorType::Sparse);           \
        default:                                                \
            throw std::invalid_argument("invalid vector type"); \
    }

#define RPY_MAKE_ALGTYPE_SWITCH(ALGTYPE)                         \
    switch (ALGTYPE) {                                           \
        case AlgebraType::FreeTensor:                            \
            return RPY_SWITCH_FN(AlgebraType::FreeTensor);       \
        case AlgebraType::Lie:                                   \
            return RPY_SWITCH_FN(AlgebraType::Lie);              \
        case AlgebraType::ShuffleTensor:                         \
            return RPY_SWITCH_FN(AlgebraType::ShuffleTensor);    \
        default:                                                 \
            throw std::invalid_argument("invalid algebra type"); \
    }

namespace rpy {
namespace algebra {

struct SignatureData;
struct DerivativeComputeInfo;
class VectorConstructionData;
class ContextBase;
class Context;

using base_context_pointer = boost::intrusive_ptr<const ContextBase>;
using context_pointer = boost::intrusive_ptr<const Context>;

struct BasicContextSpec {
    string stype_id;
    string backend;
    deg_t width;
    deg_t depth;
};


RPY_NO_DISCARD ROUGHPY_ALGEBRA_EXPORT
BasicContextSpec get_context_spec(const context_pointer & ctx);

RPY_NO_DISCARD ROUGHPY_ALGEBRA_EXPORT
context_pointer from_context_spec(const BasicContextSpec& spec);

RPY_NO_DISCARD ROUGHPY_ALGEBRA_EXPORT
std::vector<byte> alg_to_raw_bytes(context_pointer ctx, AlgebraType atype, RawUnspecifiedAlgebraType alg);

RPY_NO_DISCARD ROUGHPY_ALGEBRA_EXPORT
UnspecifiedAlgebraType alg_from_raw_bytes(context_pointer ctx, AlgebraType atype, Slice<byte> raw_data);


}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_CONTEXT_FWD_H_

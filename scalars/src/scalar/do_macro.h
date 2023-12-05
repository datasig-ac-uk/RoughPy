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
// Created by user on 01/11/23.
//

#ifndef ROUGHPY_SCALARS_SRC_SCALAR_DO_MACRO_H_
#define ROUGHPY_SCALARS_SRC_SCALAR_DO_MACRO_H_

#define DO_FOR_EACH_X(INFO)                                                    \
    switch (INFO.code) {                                                       \
        case devices::TypeCode::Int:                                           \
            switch (INFO.bytes) {                                              \
                case 1: X(int8_t);                                             \
                case 2: X(int16_t);                                            \
                case 4: X(int32_t);                                            \
                case 8: X(int64_t);                                            \
            }                                                                  \
            break;                                                             \
        case devices::TypeCode::UInt:                                          \
            switch (INFO.bytes) {                                              \
                case 1: X(uint8_t);                                             \
                case 2: X(uint16_t);                                            \
                case 4: X(uint32_t);                                            \
                case 8: X(uint64_t);                                            \
            }                                                                  \
            break;                                                             \
        case devices::TypeCode::Float:                                         \
            switch (INFO.bytes) {                                              \
                case 2: X(half);                                               \
                case 4: X(float);                                              \
                case 8: X(double);                                             \
            }                                                                  \
            break;                                                             \
        case devices::TypeCode::OpaqueHandle: break;                           \
        case devices::TypeCode::BFloat:                                        \
            if (INFO.bytes == 2) { X(bfloat16); }                              \
            break;                                                             \
        case devices::TypeCode::Complex: break;                                \
        case devices::TypeCode::Bool: break;                                   \
        case devices::TypeCode::Rational: break;                               \
        case devices::TypeCode::ArbitraryPrecision: break;                     \
        case devices::TypeCode::ArbitraryPrecisionUInt: break;                 \
        case devices::TypeCode::ArbitraryPrecisionFloat: break;                \
        case devices::TypeCode::ArbitraryPrecisionComplex: break;              \
        case devices::TypeCode::ArbitraryPrecisionRational:                    \
            X(rational_scalar_type);                                           \
        case devices::TypeCode::APRationalPolynomial: X(rational_poly_scalar); \
    }


#endif// ROUGHPY_SCALARS_SRC_SCALAR_DO_MACRO_H_

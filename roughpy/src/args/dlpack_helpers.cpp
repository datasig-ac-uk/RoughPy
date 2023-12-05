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
// Created by sam on 11/08/23.
//

#include "dlpack_helpers.h"

#include <roughpy/scalars/scalar_types.h>

using namespace rpy;
using namespace python;



#define DLTHROW(type, bits) \
    RPY_THROW(std::runtime_error, \
              std::to_string(bits) + " bit " #type " is not supported")
//
//const string&
//python::type_id_for_dl_info(const DLDataType& dtype, const DLDevice& device)
//{
//    if (device.device_type != kDLCPU) {
//        RPY_THROW(
//                std::runtime_error,
//                "for the time being, constructing non-cpu "
//                "devices are not supported by RoughPy"
//        );
//    }
//
//    switch (dtype.code) {
//        case kDLFloat:
//            switch (dtype.bits) {
//                case 16: return scalars::type_id_of<scalars::half>();
//                case 32: return scalars::type_id_of<float>();
//                case 64: return scalars::type_id_of<double>();
//                default: DLTHROW(float, dtype.bits);
//            }
//        case kDLInt:
//            switch (dtype.bits) {
//                case 8: return scalars::type_id_of<char>();
//                case 16: return scalars::type_id_of<short>();
//                case 32: return scalars::type_id_of<int>();
//                case 64: return scalars::type_id_of<long long>();
//                default: DLTHROW(int, dtype.bits);
//            }
//        case kDLUInt:
//            switch (dtype.bits) {
//                case 8: return scalars::type_id_of<unsigned char>();
//                case 16: return scalars::type_id_of<unsigned short>();
//                case 32: return scalars::type_id_of<unsigned int>();
//                case 64: return scalars::type_id_of<unsigned long long>();
//                default: DLTHROW(uint, dtype.bits);
//            }
//        case kDLBfloat:
//            if (dtype.bits == 16) {
//                return scalars::type_id_of<scalars::bfloat16>();
//            } else {
//                DLTHROW(bfloat, dtype.bits);
//            }
//        case kDLComplex: DLTHROW(complex, dtype.bits);
//        case kDLOpaqueHandle: DLTHROW(opaquehandle, dtype.bits);
//        case kDLBool: DLTHROW(bool, dtype.bits);
//    }
//    RPY_UNREACHABLE_RETURN({});
//}

#undef DLTHROW



const scalars::ScalarType*
python::scalar_type_for_dl_info(const DLDataType& dtype, const DLDevice& device)
{
    auto tp = scalars::scalar_type_of(convert_from_dl_datatype(dtype));
    RPY_CHECK(tp);
    return *tp;
}

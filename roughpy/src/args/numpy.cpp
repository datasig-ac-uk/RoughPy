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

#include "numpy.h"

using namespace rpy;

namespace {

enum NpyDtypes : int
{
    NPY_BOOL = 0,
    NPY_BYTE,
    NPY_UBYTE,
    NPY_SHORT,
    NPY_USHORT,
    NPY_INT,
    NPY_UINT,
    NPY_LONG,
    NPY_ULONG,
    NPY_LONGLONG,
    NPY_ULONGLONG,
    NPY_FLOAT,
    NPY_DOUBLE,
    NPY_LONGDOUBLE,
    NPY_CFLOAT,
    NPY_CDOUBLE,
    NPY_CLONGDOUBLE,
    NPY_OBJECT = 17,
    NPY_STRING,
    NPY_UNICODE,
    NPY_VOID,
    /*
     * New 1.6 types appended, may be integrated
     * into the above in 2.0.
     */
    NPY_DATETIME,
    NPY_TIMEDELTA,
    NPY_HALF,

    NPY_NTYPES,
    NPY_NOTYPE,
    NPY_CHAR,
    NPY_USERDEF = 256, /* leave room for characters */

    /* The number of types not including the new 1.6 types */
    NPY_NTYPES_ABI_COMPATIBLE = 21
};

}// namespace

const scalars::ScalarType* python::npy_dtype_to_ctype(pybind11::dtype dtype)
{
    const scalars::ScalarType* type = nullptr;

    switch (dtype.num()) {
        case NPY_FLOAT: type = scalars::ScalarType::of<float>(); break;
        case NPY_DOUBLE: type = scalars::ScalarType::of<double>(); break;
        default:
            // Default behaviour, promote to double
            type = scalars::ScalarType::of<double>();
            break;
    }

    return type;
}
pybind11::dtype python::ctype_to_npy_dtype(const scalars::ScalarType* type)
{
    if (type == scalars::ScalarType::of<double>()) { return py::dtype("d"); }
    if (type == scalars::ScalarType::of<float>()) { return py::dtype("f"); }

    RPY_THROW(py::type_error, "unsupported data type");
}

string python::npy_dtype_to_identifier(pybind11::dtype dtype)
{
    string identifier;

    switch (dtype.num()) {
        case NPY_FLOAT: identifier = "f32"; break;
        case NPY_DOUBLE: identifier = "f64"; break;
        case NPY_INT: identifier = "i32"; break;
        case NPY_UINT: identifier = "u32"; break;
        case NPY_LONG: {
            if (sizeof(long) == sizeof(int)) {
                identifier = "i32";
            } else {
                identifier = "i64";
            }
        } break;
        case NPY_ULONG: {
            if (sizeof(long) == sizeof(int)) {
                identifier = "ui32";
            } else {
                identifier = "u64";
            }
        } break;
        case NPY_LONGLONG: identifier = "i64"; break;
        case NPY_ULONGLONG: identifier = "u64"; break;

        case NPY_BOOL:
        case NPY_BYTE: identifier = "i8"; break;
        case NPY_UBYTE: identifier = "u8"; break;
        case NPY_SHORT: identifier = "i16"; break;
        case NPY_USHORT: identifier = "u16"; break;

        default: RPY_THROW(py::type_error, "unsupported dtype");
    }

    return identifier;
}

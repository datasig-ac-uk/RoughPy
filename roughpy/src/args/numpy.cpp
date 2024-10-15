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
#include <roughpy/platform/devices/core.h>
#include <roughpy/scalars/scalar_type.h>
#include <roughpy/scalars/scalar_types.h>
#include <roughpy/scalars/traits.h>

#define NPY_NO_DEPRECATED_API NPY_1_17_API_VERSION
#include <numpy/arrayobject.h>

#include "scalars/r_py_polynomial.h"
#include "scalars/scalars.h"

#include <sstream>

using namespace rpy;

// namespace {
//
// enum NpyDtypes : int
// {
//     NPY_BOOL = 0,
//     NPY_BYTE,
//     NPY_UBYTE,
//     NPY_SHORT,
//     NPY_USHORT,
//     NPY_INT,
//     NPY_UINT,
//     NPY_LONG,
//     NPY_ULONG,
//     NPY_LONGLONG,
//     NPY_ULONGLONG,
//     NPY_FLOAT,
//     NPY_DOUBLE,
//     NPY_LONGDOUBLE,
//     NPY_CFLOAT,
//     NPY_CDOUBLE,
//     NPY_CLONGDOUBLE,
//     NPY_OBJECT = 17,
//     NPY_STRING,
//     NPY_UNICODE,
//     NPY_VOID,
//     /*
//      * New 1.6 types appended, may be integrated
//      * into the above in 2.0.
//      */
//     NPY_DATETIME,
//     NPY_TIMEDELTA,
//     NPY_HALF,
//
//     NPY_NTYPES,
//     NPY_NOTYPE,
//     NPY_CHAR,
//     NPY_USERDEF = 256, /* leave room for characters */
//
//     /* The number of types not including the new 1.6 types */
//     NPY_NTYPES_ABI_COMPATIBLE = 21
// };
//
// }// namespace

static inline int info_to_typenum(const devices::TypeInfo& info)
{
    switch (info.code) {
        case devices::TypeCode::Int:
            switch (info.bytes) {
                case 1: return NPY_INT8;
                case 2: return NPY_INT16;
                case 4: return NPY_INT32;
                case 8: return NPY_INT64;
                default: break;
            }
            break;
        case devices::TypeCode::UInt:
            switch (info.bytes) {
                case 1: return NPY_UINT8;
                case 2: return NPY_UINT16;
                case 4: return NPY_UINT32;
                case 8: return NPY_UINT64;
                default: break;
            }
            break;
        case devices::TypeCode::Float:
            switch (info.bytes) {
                case 2: return NPY_FLOAT16;
                case 4: return NPY_FLOAT32;
                case 8: return NPY_FLOAT64;
                default: break;
            }
            break;
        case devices::TypeCode::OpaqueHandle: break;
        case devices::TypeCode::BFloat: break;
        case devices::TypeCode::Complex: break;
        case devices::TypeCode::Bool: break;
        case devices::TypeCode::Rational:
        case devices::TypeCode::ArbitraryPrecisionInt:
        case devices::TypeCode::ArbitraryPrecisionUInt:
        case devices::TypeCode::ArbitraryPrecisionFloat:
        case devices::TypeCode::ArbitraryPrecisionComplex:
        case devices::TypeCode::ArbitraryPrecisionRational:
        case devices::TypeCode::APRationalPolynomial: return NPY_OBJECT;
    }

    std::stringstream ss;
    ss << "scalar type " << info << " is not supported in conversions";

    RPY_THROW(std::runtime_error, ss.str());
}

const scalars::ScalarType* python::npy_dtype_to_ctype(pybind11::dtype dtype)
{
    const scalars::ScalarType* type = nullptr;

    switch (dtype.num()) {
        case NPY_FLOAT: type = *scalars::ScalarType::of<float>(); break;
        case NPY_DOUBLE: type = *scalars::ScalarType::of<double>(); break;
        default:
            // Default behaviour, promote to double
            type = *scalars::ScalarType::of<double>();
            break;
    }

    return type;
}

pybind11::dtype python::ctype_to_npy_dtype(const scalars::ScalarType* type)
{
    return py::dtype(info_to_typenum(type->type_info()));
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

namespace {

template <typename T>
void write_type_as_py_object(PyObject** dst, Slice<const T> data)
{
    const auto type_o = scalars::scalar_type_of<T>();
    RPY_CHECK(type_o);
    for (dimn_t i = 0; i < data.size(); ++i) {
        Py_XDECREF(dst[i]);
        dst[i] = py::cast(scalars::Scalar(*type_o, data[i])).release().ptr();
    }
}

void write_type_as_py_object(
        PyObject** dst,
        Slice<const scalars::rational_poly_scalar> data
)
{
    for (dimn_t i = 0; i < data.size(); ++i) {
        Py_XDECREF(dst[i]);
        dst[i] = PyPolynomial_FromPolynomial(
                scalars::rational_poly_scalar(data[i])
        );
    }
}

}// namespace

py::array python::dtl::dense_data_to_array(
        const scalars::ScalarArray& data,
        dimn_t dimension
)
{
    RPY_DBG_ASSERT(data.size() <= dimension);
    const auto type_info = data.type_info();

    py::dtype dtype(info_to_typenum(type_info));

    auto result
            = python::dtl::new_zero_array_for_stype(*data.type(), dimension);

    if (scalars::traits::is_fundamental(type_info)) {
        scalars::dtl::scalar_convert_copy(
                result.mutable_data(),
                type_info,
                data.pointer(),
                type_info,
                data.size()
        );
    } else {
        PyArray_FillWithScalar(
                reinterpret_cast<PyArrayObject*>(result.ptr()),
                Py_None
        );

        auto** raw = static_cast<PyObject**>(result.mutable_data());

        switch (type_info.code) {
            case devices::TypeCode::ArbitraryPrecisionRational:
                write_type_as_py_object(
                        raw,
                        data.template as_slice<devices::rational_scalar_type>()
                );
                break;
            case devices::TypeCode::APRationalPolynomial:
                write_type_as_py_object(
                        raw,
                        data.template as_slice<devices::rational_poly_scalar>()
                );
                break;
            case devices::TypeCode::Int:
            case devices::TypeCode::UInt:
            case devices::TypeCode::Float:
            case devices::TypeCode::OpaqueHandle:
            case devices::TypeCode::BFloat:
            case devices::TypeCode::Complex:
            case devices::TypeCode::Bool:
            case devices::TypeCode::Rational:
            case devices::TypeCode::ArbitraryPrecisionInt:
            case devices::TypeCode::ArbitraryPrecisionUInt:
            case devices::TypeCode::ArbitraryPrecisionFloat:
            case devices::TypeCode::ArbitraryPrecisionComplex:
                RPY_THROW(
                        std::runtime_error,
                        "this case should have been handled elsewhere"
                );
        }
    }

    return result;
}

py::array python::dtl::new_zero_array_for_stype(
        const scalars::ScalarType* type,
        dimn_t dimension
)
{
    auto typenum = info_to_typenum(type->type_info());
    py::dtype dtype(typenum);
    py::array result(dtype, {static_cast<py::ssize_t>(dimension)}, {});

    if (typenum == NPY_OBJECT) {
        PyArray_FillWithScalar(
                reinterpret_cast<PyArrayObject*>(result.ptr()),
                Py_None
        );
    } else {
        PyArray_FILLWBYTE(reinterpret_cast<PyArrayObject*>(result.ptr()), 0);
    }

    return result;
}

void python::dtl::write_entry_to_array(
        py::array& array,
        dimn_t index,
        const scalars::Scalar& arg
)
{}

static inline PyTypeObject* import_numpy_impl() noexcept
{
    import_array();
    return &PyArray_Type;
}

void python::import_numpy()
{
#ifdef ROUGHPY_WITH_NUMPY
    if (import_numpy_impl() == nullptr) { throw py::error_already_set(); }
#endif
}

bool python::dtl::is_object_dtype(py::dtype dtype) noexcept
{
    return dtype.num() == NPY_OBJECT;
}

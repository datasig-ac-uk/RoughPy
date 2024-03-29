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

#ifndef RPY_PY_SCALARS_SCALARS_H_
#define RPY_PY_SCALARS_SCALARS_H_

#include "roughpy_module.h"

#include <functional>

#include <roughpy/scalars/key_scalar_array.h>
#include <roughpy/scalars/scalar_type.h>

#include "r_py_polynomial.h"

#include "dlpack.h"

namespace rpy {
namespace python {

struct RPY_NO_EXPORT AlternativeKeyType {
    py::handle py_key_type;
    std::function<key_type(py::handle)> converter;
};

struct ArgSizeInfo {
    idimn_t num_values;
    idimn_t num_keys;
};

enum class GroundDataType
{
    UnSet,
    Scalars,
    KeyValuePairs
};

struct RPY_NO_EXPORT PyToBufferOptions {
    /// Scalar type to use. If null, will be set to the resulting type
    const scalars::ScalarType* type = nullptr;

    /// Maximum number of nested objects to search. Set to 0 for no recursion.
    dimn_t max_nested = 0;

    /// Information about the constructed array
    std::vector<idimn_t> shape;

    /// Allow a single, untagged scalar as argument
    bool allow_scalar = true;

    /// Do not check std library types or imported data types.
    /// All Python types will (try) to be converted to double.
    bool no_check_imported = false;

    /// Alternative acceptable key_type/conversion pair
    AlternativeKeyType* alternative_key = nullptr;
};

inline py::type get_py_rational()
{
    return py::reinterpret_borrow<py::type>(
            py::module_::import("fractions").attr("Fraction")
    );
}

inline py::type get_py_decimal()
{
    return py::reinterpret_borrow<py::type>(
            py::module_::import("decimal").attr("Decimal")
    );
}

inline bool is_scalar(py::handle arg)
{
    return (py::isinstance<py::int_>(arg) || py::isinstance<py::float_>(arg)
            || RPyPolynomial_Check(arg.ptr()));
}

inline bool is_key(py::handle arg, python::AlternativeKeyType* alternative)
{
    if (alternative != nullptr) {
        return py::isinstance<py::int_>(arg)
                || py::isinstance(arg, alternative->py_key_type);
    }
    if (py::isinstance<py::int_>(arg)) { return true; }
    return false;
}

inline bool is_kv_pair(py::handle arg, python::AlternativeKeyType* alternative)
{
    if (py::isinstance<py::tuple>(arg)) {
        auto tpl = py::reinterpret_borrow<py::tuple>(arg);
        if (tpl.size() == 2) { return is_key(tpl[0], alternative); }
    }
    return false;
}


scalars::KeyScalarArray
py_to_buffer(const py::handle& arg, PyToBufferOptions& options);

void assign_py_object_to_scalar(scalars::Scalar& dst, py::handle object);

scalars::Scalar
py_to_scalar(const scalars::ScalarType* type, py::handle object);

ArgSizeInfo compute_size_and_type(
        python::PyToBufferOptions& options, std::vector<py::object>& leaves,
        py::handle arg
);



void init_scalars(py::module_& m);

}// namespace python
}// namespace rpy

#endif// RPY_PY_SCALARS_SCALARS_H_

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

//
// Created by user on 15/03/23.
//

#ifndef ROUGHPY_ROUGHPY_SRC_ALGEBRA_SETUP_ALGEBRA_TYPE_H
#define ROUGHPY_ROUGHPY_SRC_ALGEBRA_SETUP_ALGEBRA_TYPE_H

#include "roughpy_module.h"

#include <sstream>

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <roughpy/algebra/algebra_fwd.h>
#include <roughpy/scalars/scalar.h>

#include "args/numpy.h"
#include "scalars/scalar_type.h"

namespace rpy {
namespace python {

/**
 * @brief Set up a new algebra python object.
 * @tparam Alg Algebra type to set up
 * @tparam Args Additional args (deduced)
 * @param klass pybind11 class_ instance to set up
 */
template <typename Alg, typename... Args>
void setup_algebra_type(py::class_<Alg, Args...>& klass)
{

    using namespace pybind11::literals;

    py::options options;
    options.disable_function_signatures();

    // setup static properties
    /*
     * Use lambda wrapping functions because there is an issue with "noexcept"
     * modifier on these methods in C++17 mode.
     */
    klass.def_property_readonly("width", [](const Alg& arg) {
        return arg.width();
    });
    klass.def_property_readonly("max_degree", [](const Alg& arg) {
        return arg.depth();
    });
    klass.def_property_readonly("dtype", [](const Alg& arg) {
        return to_ctype_type(arg.coeff_type());
    });
    klass.def_property_readonly("storage_type", [](const Alg& arg) {
        return arg.storage_type();
    });

    // setup dynamic properties
    klass.def("size", &Alg::size);
    klass.def("dimension", &Alg::dimension);
    klass.def("degree", &Alg::degree);

    klass.def("is_zero", &Alg::is_zero);
    // TODO: Add access and iteration methods

    klass.def(
            "__iter__",
            [](const Alg& self) {
                return py::make_iterator(self.begin(), self.end());
            },
            py::keep_alive<0, 1>()
    );

    // setup arithmetic
    klass.def("__neg__", &Alg::uminus, py::is_operator());

    klass.def("__add__", &Alg::add, py::is_operator());
    klass.def("__sub__", &Alg::sub, py::is_operator());
    klass.def("__mul__", &Alg::smul, py::is_operator());
    klass.def("__truediv__", &Alg::smul, py::is_operator());
    klass.def("__mul__", &Alg::mul, py::is_operator());
    klass.def(
            "__rmul__",
            [](const Alg& self, const scalars::Scalar& other) {
                return self.smul(other);
            },
            py::is_operator()
    );
    klass.def(
            "__mul__",
            [](const Alg& self, scalar_t arg) {
                return self.smul(scalars::Scalar(arg));
            },
            py::is_operator()
    );
    klass.def(
            "__rmul__",
            [](const Alg& self, scalar_t arg) {
                return self.smul(scalars::Scalar(arg));
            },
            py::is_operator()
    );
    klass.def(
            "__mul__",
            [](const Alg& self, long long arg) {
                return self.smul(scalars::Scalar(self.coeff_type(), arg, 1LL));
            },
            py::is_operator()
    );
    klass.def(
            "__rmul__",
            [](const Alg& self, long long arg) {
                return self.smul(scalars::Scalar(self.coeff_type(), arg, 1LL));
            },
            py::is_operator()
    );
    klass.def(
            "__truediv__",
            [](const Alg& self, scalar_t arg) {
                return self.sdiv(scalars::Scalar(arg));
            },
            py::is_operator()
    );
    klass.def(
            "__truediv__",
            [](const Alg& self, long long arg) {
                return self.sdiv(scalars::Scalar(self.coeff_type(), arg, 1LL));
            },
            py::is_operator()
    );

    klass.def("__iadd__", &Alg::add_inplace, py::is_operator());
    klass.def("__isub__", &Alg::sub_inplace, py::is_operator());
    klass.def("__imul__", &Alg::smul_inplace, py::is_operator());
    klass.def("__itruediv__", &Alg::sdiv_inplace, py::is_operator());
    klass.def("__imul__", &Alg::mul_inplace, py::is_operator());

    klass.def(
            "__imul__",
            [](Alg& self, scalar_t arg) {
                return self.smul_inplace(scalars::Scalar(arg));
            },
            py::is_operator()
    );
    klass.def(
            "__imul__",
            [](Alg& self, long long arg) {
                return self.smul_inplace(
                        scalars::Scalar(self.coeff_type(), arg, 1LL)
                );
            },
            py::is_operator()
    );
    klass.def(
            "__itruediv__",
            [](Alg& self, scalar_t arg) {
                return self.sdiv_inplace(scalars::Scalar(arg));
            },
            py::is_operator()
    );
    klass.def(
            "__itruediv__",
            [](Alg& self, long long arg) {
                return self.sdiv_inplace(
                        scalars::Scalar(self.coeff_type(), arg, 1LL)
                );
            },
            py::is_operator()
    );

    // setup fused inplace ops
    klass.def("add_scal_mul", &Alg::add_scal_mul, "other"_a, "scalar"_a);
    klass.def("sub_scal_mul", &Alg::sub_scal_mul, "other"_a, "scalar"_a);
    klass.def("add_scal_div", &Alg::add_scal_div, "other"_a, "scalar"_a);
    klass.def("sub_scal_div", &Alg::sub_scal_div, "other"_a, "scalar"_a);

    klass.def("add_mul", &Alg::add_mul, "lhs"_a, "rhs"_a);
    klass.def("sub_mul", &Alg::sub_mul, "lhs"_a, "rhs"_a);
    klass.def("mul_smul", &Alg::mul_smul, "other"_a, "scalar"_a);
    klass.def("mul_sdiv", &Alg::mul_sdiv, "other"_a, "scalar"_a);

    // setup string function
    klass.def("__str__", [](const Alg& self) {
        std::stringstream ss;
        self.print(ss);
        return ss.str();
    });

    // setup comparisons
    klass.def(py::self == py::self);
    klass.def(py::self != py::self);

    // setup conversion to numpy array
#ifdef ROUGHPY_WITH_NUMPY
    klass.def("__array__", [](const Alg& self) {
        //        py::dtype dtype = dtype_from(self.coeff_type());
        py::dtype dtype = ctype_to_npy_dtype(self.coeff_type());

        auto dense_data = self.dense_data();
        if (dense_data) {
            const auto dense_data_inner = *dense_data;
            return py::array(
                    dtype, {dense_data_inner.size()}, {}, dense_data_inner.ptr()
            );
        }
        return py::array(dtype);
    });
#endif

    // TODO: DLpack interface
}

}// namespace python
}// namespace rpy

#endif// ROUGHPY_ROUGHPY_SRC_ALGEBRA_SETUP_ALGEBRA_TYPE_H

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

#ifndef RPY_PY_ARGS_NUMPY_H_
#define RPY_PY_ARGS_NUMPY_H_
#ifdef ROUGHPY_WITH_NUMPY

#  include "roughpy_module.h"
#  include <pybind11/numpy.h>

#  include <roughpy/scalars.h>
#include <roughpy/algebra/algebra_base.h>

#  include <string>

namespace rpy {
namespace python {

const scalars::ScalarType* npy_dtype_to_ctype(py::dtype dtype);

py::dtype ctype_to_npy_dtype(const scalars::ScalarType* type);

string npy_dtype_to_identifier(py::dtype dtype);

namespace dtl {

RPY_NO_DISCARD
py::array dense_data_to_array(const scalars::ScalarArray& data,
                              dimn_t dimension);
RPY_NO_DISCARD
py::array new_zero_array_for_stype(const scalars::ScalarType* type,
                                   dimn_t dimension);
void write_entry_to_array(py::array& array,
                          dimn_t index,
                          const scalars::Scalar& arg);

bool is_object_dtype(py::dtype dtype) noexcept;
}

template <typename Interface, template <typename, template <typename> class>
          class DerivedImpl>
RPY_NO_DISCARD
inline py::array algebra_to_array(
    const algebra::AlgebraBase<Interface, DerivedImpl>& alg, bool copy)
{
    const auto* stype = alg.coeff_type();
    const auto basis = alg.basis();
    const auto dimension = basis.dimension();

    auto dense_data = alg.dense_data();
    copy |= dtl::is_object_dtype(ctype_to_npy_dtype(stype));

    if (!copy && dense_data && dense_data->size() == dimension) {
        // Dense and full dimension, borrow
        auto dtype = ctype_to_npy_dtype(stype);
        return py::array(dtype, {dimension}, {}, dense_data->pointer());
    }
    if (dense_data) { return dtl::dense_data_to_array(*dense_data, dimension); }

    auto result = dtl::new_zero_array_for_stype(stype, dimension);

    for (auto&& item : alg) {
        dimn_t index = basis.key_to_index(item.key());
        dtl::write_entry_to_array(result, index, item.value());
    }

    return result;
}

}// namespace python
}// namespace rpy

#endif// ROUGHPY_WITH_NUMPY

namespace rpy {
namespace python {

void import_numpy();

}
}


#endif// RPY_PY_ARGS_NUMPY_H_

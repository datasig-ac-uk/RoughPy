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

#include "basis.h"

#include <roughpy/algebra/basis.h>
#include <roughpy/algebra/lie_basis.h>
#include <roughpy/algebra/tensor_basis.h>

#include "lie_key.h"
#include "lie_key_iterator.h"
#include "tensor_key.h"
#include "tensor_key_iterator.h"

using namespace rpy;
using namespace rpy::algebra;
using namespace pybind11::literals;

template <typename T, typename K, typename KIter>
static py::class_<T> wordlike_basis_setup(py::module_& m, const char* name)
{

    py::class_<T> basis(m, name);

//    basis.def(py::init<deg_t, deg_t>([](deg_t width, deg_t depth) {
//
//         }));

    basis.def_property_readonly("width", &T::width);
    basis.def_property_readonly("depth", &T::depth);
    basis.def_property_readonly("dimension", &T::dimension);

    basis.def(
            "index_to_key",
            [](const T& self, dimn_t index) {
                return K(self, self.index_to_key(index));
            },
            "index"_a
    );
    basis.def(
            "key_to_index",
            [](const T& self, const python::PyLieKey& key) {
                return self.key_to_index(0);
            },
            "key"_a
    );

    basis.def(
            "parents",
            [](const T& self, const K& key) { return self.parents(0); },
            "key"_a
    );
    basis.def("size", &T::size);

    basis.def("__iter__", [](const T& self) {
        return KIter(self);
    });

    return basis;
}

void python::init_basis(py::module_& m)
{

    wordlike_basis_setup<TensorBasis, PyTensorKey, PyTensorKeyIterator>(
            m, "TensorBasis"
    );

    wordlike_basis_setup<LieBasis, PyLieKey, PyLieKeyIterator>(
            m, "LieBasis"
    );
}

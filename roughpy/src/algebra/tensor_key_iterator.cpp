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

#include "tensor_key_iterator.h"

#include <roughpy/algebra/basis.h>
#include <roughpy/algebra/context.h>

using namespace rpy;
using namespace pybind11::literals;

static const char* TKEY_ITERATOR_DOC = R"eadoc(Iterator over tensor words.
)eadoc";

python::PyTensorKeyIterator::PyTensorKeyIterator(
        algebra::TensorBasis basis, key_type current, key_type end
)
    : m_current(current), m_end(end), m_basis(basis)
{
    auto dim = m_basis.dimension();
    if (m_end >= dim) {
        m_end = dim;
    }
}
python::PyTensorKey python::PyTensorKeyIterator::next()
{
    if (m_current >= m_end) { throw py::stop_iteration(); }
    auto current = m_current;
    ++m_current;
    return PyTensorKey(m_basis, current);
}

void python::init_tensor_key_iterator(py::module_& m)
{

    py::class_<PyTensorKeyIterator> klass(
            m, "TensorKeyIterator", TKEY_ITERATOR_DOC
    );

    klass.def(
            py::init([](const PyTensorKey& start_key) {
                return PyTensorKeyIterator(
                        start_key.basis(),
                        static_cast<key_type>(start_key)
                );
            }),
            "start_key"_a
    );
    klass.def(
            py::init([](const PyTensorKey& start_key,
                        const PyTensorKey& end_key) {
                return PyTensorKeyIterator(
                        start_key.basis(),
                        static_cast<key_type>(start_key),
                        static_cast<key_type>(end_key)
                );
            }),
            "start_key"_a, "end_key"_a
    );
    klass.def("__iter__", [](PyTensorKeyIterator& self) { return self; });
    klass.def("__next__", &PyTensorKeyIterator::next);
}

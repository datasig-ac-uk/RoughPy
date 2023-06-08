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

#include "lie_key_iterator.h"

#include <roughpy/algebra/basis.h>
#include <roughpy/algebra/context.h>

#include "context.h"


using namespace rpy;
using namespace pybind11::literals;


static const char *LKEY_ITERATOR_DOC = R"eadoc(Iterator over range of Hall set members.
)eadoc";

python::PyLieKeyIterator::PyLieKeyIterator(const py::object& ctx, key_type current, key_type end)
    : m_current(current), m_end(end)
{
    if (!py::isinstance(ctx, reinterpret_cast<PyObject*>(&RPyContext_Type))) {
        throw py::type_error("expected a Context object");
    }
    p_ctx = python::ctx_cast(ctx.ptr());
}

static python::PyLieKey to_py_lie_key(key_type k, const algebra::LieBasis &lbasis) {
    auto width = lbasis.width();

    if (lbasis.letter(k)) {
        return python::PyLieKey(width, k);
    }

    auto lparent = *lbasis.lparent(k);
    auto rparent = *lbasis.rparent(k);

    if (lbasis.letter(lparent) && lbasis.letter(rparent)) {
        return python::PyLieKey(lbasis.width(), lparent, rparent);
    }
    if (lbasis.letter(lparent)) {
        return python::PyLieKey(width, lparent, to_py_lie_key(rparent, lbasis));
    }
    return python::PyLieKey(width,
                      to_py_lie_key(lparent, lbasis),
                      to_py_lie_key(rparent, lbasis));
}

python::PyLieKey python::PyLieKeyIterator::next() {
    if (m_current > m_end) {
        throw py::stop_iteration();
    }
    auto current = m_current;
    ++m_current;
    return to_py_lie_key(current, p_ctx->get_lie_basis());
}

void python::init_lie_key_iterator(py::module_ &m) {
    py::class_<PyLieKeyIterator> klass(m, "LieKeyIterator", LKEY_ITERATOR_DOC);
    klass.def(py::init<py::object>(), "context"_a);
    klass.def(py::init<py::object, key_type>(), "context"_a, "start_key"_a);
    klass.def(py::init<py::object, key_type, key_type>(), "context"_a, "start_key"_a, "end_key"_a);

    klass.def("__iter__", [](PyLieKeyIterator &self) { return self; });
    klass.def("__next__", &PyLieKeyIterator::next);
}

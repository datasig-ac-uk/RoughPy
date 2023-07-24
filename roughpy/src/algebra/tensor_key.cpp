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

#include "tensor_key.h"
#include <roughpy/algebra/context.h>

#include <algorithm>
#include <sstream>

using namespace rpy;
using namespace pybind11::literals;


python::PyTensorKey::PyTensorKey(algebra::TensorBasis basis, key_type key)
    : m_key(key), m_basis(move(basis))
{}

python::PyTensorKey::operator key_type() const noexcept { return m_key; }
string python::PyTensorKey::to_string() const
{
    std::stringstream ss;
    ss << '(';
    bool not_first = false;
    for (auto letter : to_letters()) {
        if (not_first) { ss << ','; }
        ss << letter;
        not_first = true;
    }
    ss << ')';
    return ss.str();
}
python::PyTensorKey python::PyTensorKey::lparent() const
{
    return python::PyTensorKey(m_basis, 0);
}
python::PyTensorKey python::PyTensorKey::rparent() const
{
    return python::PyTensorKey(m_basis, 0);
}
pair<python::PyTensorKey, python::PyTensorKey>
python::PyTensorKey::split_n(deg_t n) const
{
    if (m_key == 0) {
        return std::make_pair(PyTensorKey(m_basis, 0), PyTensorKey(m_basis, 0));
    }
    if (n <= 0) {
        return std::make_pair(*this,PyTensorKey(m_basis, 0));
    }
    if (n >= m_basis.depth()) {
        return std::make_pair(PyTensorKey(m_basis, 0), *this);
    }

    auto width = m_basis.width();
    //TODO: needs better implementation
    key_type shift = 1;
    for (deg_t i=0; i<n; ++i) {
        shift *= width;
    }
    auto left = (m_key - 1) / shift;
    auto right = 1 + ((m_key - 1) % shift);

    return std::make_pair(PyTensorKey(m_basis, left), PyTensorKey(m_basis, right));

}
bool python::PyTensorKey::is_letter() const
{
    return 1 <= m_key && m_key <= static_cast<key_type>(m_basis.width());
}
deg_t python::PyTensorKey::width() const { return m_basis.width(); }
deg_t python::PyTensorKey::depth() const { return m_basis.depth(); }
deg_t python::PyTensorKey::degree() const { return m_basis.depth(); }
std::vector<let_t> python::PyTensorKey::to_letters() const
{
    std::vector<let_t> letters;
    const auto width = m_basis.width();
    const auto depth = m_basis.depth();
    letters.reserve(depth);
    auto tmp = m_key;
    while (tmp) {
        tmp -= 1;
        letters.push_back(1 + (tmp % width));
        tmp /= width;
    }
    std::reverse(letters.begin(), letters.end());
    return letters;
}
bool python::PyTensorKey::equals(const python::PyTensorKey& other
) const noexcept
{
    return m_basis.width() == other.m_basis.width() && m_key == other.m_key;
}
bool python::PyTensorKey::less(const python::PyTensorKey& other) const noexcept
{
    return m_key < other.m_key;
}

static python::PyTensorKey
construct_key(const py::args& args, const py::kwargs& kwargs)
{
    std::vector<let_t> letters;

    if (args.empty() && kwargs.contains("index")) {
        auto width = kwargs["width"].cast<deg_t>();
        auto depth = kwargs["depth"].cast<deg_t>();
        auto index = kwargs["index"].cast<key_type>();

        auto max_idx = (python::maths::power(dimn_t(width), depth + 1) - 1)
                / (dimn_t(width) - 1);
        if (index >= max_idx) {
            RPY_THROW(py::value_error, "provided index exceeds maximum");
        }

        auto ctx = algebra::get_context(width, depth,
                                        scalars::ScalarType::of<float>());
        return python::PyTensorKey(ctx->get_tensor_basis(), index);
    }

    if (!args.empty() && py::isinstance<py::sequence>(args[0])) {
        letters.reserve(py::len(args[0]));
        for (auto arg : args[0]) { letters.push_back(arg.cast<let_t>()); }
    } else {
        letters.reserve(py::len(args));
        for (auto arg : args) { letters.push_back(arg.cast<let_t>()); }
    }

    deg_t width = 0;
    deg_t depth = deg_t(letters.size());

    auto max_elt = std::max_element(letters.begin(), letters.end());
    if (kwargs.contains("width")) {
        width = kwargs["width"].cast<deg_t>();
    } else if (!letters.empty()) {
        width = *max_elt;
    }

    if (kwargs.contains("depth")) { depth = kwargs["depth"].cast<deg_t>(); }

    if (letters.size() > static_cast<dimn_t>(depth)) {
        RPY_THROW(py::value_error, "number of letters exceeds specified depth");
    }

    if (!letters.empty() && *max_elt > width) {
        RPY_THROW(py::value_error, "letter value exceeds alphabet size");
    }

    key_type result = 0;
    auto wwidth = dimn_t(width);
    for (auto letter : letters) {
        result *= wwidth;
        result += key_type(letter);
    }

    auto ctx = algebra::get_context(
            width, depth, scalars::ScalarType::of<float>()
    );
    return python::PyTensorKey(ctx->get_tensor_basis(), result);
}

void python::init_py_tensor_key(py::module_& m)
{
    py::class_<PyTensorKey> klass(m, "TensorKey");
    klass.def(py::init(&construct_key));

    klass.def_property_readonly("width", &PyTensorKey::width);
    klass.def_property_readonly("max_degree", &PyTensorKey::depth);

    klass.def("to_index", [](const PyTensorKey& key) {
        return static_cast<key_type>(key);
    });
    klass.def("degree", [](const PyTensorKey& key) {
        return key.to_letters().size();
    });
    klass.def("split_n", &PyTensorKey::split_n, "n"_a);

    klass.def("__str__", &PyTensorKey::to_string);
    klass.def("__repr__", &PyTensorKey::to_string);

    klass.def("__eq__", &PyTensorKey::equals);
}

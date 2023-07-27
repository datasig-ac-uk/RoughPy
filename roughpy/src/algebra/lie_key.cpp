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

#include "lie_key.h"

#include <algorithm>
#include <cassert>
#include <sstream>

#include <roughpy/algebra/context.h>

using namespace rpy;

template <typename LeftFn, typename RightFn>
static void walk_tree(
        const python::PyLieLetter* tree, LeftFn left_visitor,
        RightFn right_visitor
)
{
    const auto* left = tree;
    const auto* right = ++tree;

    left_visitor(*left);
    right_visitor(*right);
    if (left->is_offset()) {
        walk_tree(
                left + static_cast<dimn_t>(*left), left_visitor, right_visitor
        );
    }
    if (right->is_offset()) {
        walk_tree(
                right + static_cast<dimn_t>(*right), left_visitor, right_visitor
        );
    }
}

template <typename Fn>
static void walk_tree(const python::PyLieLetter* tree, Fn visitor)
{
    walk_tree(tree, visitor, visitor);
}

static bool
branches_equal(const python::PyLieLetter* lhs, const python::PyLieLetter* rhs)
{
    if (!lhs->is_offset() && !rhs->is_offset()) {
        return static_cast<let_t>(*lhs) == static_cast<let_t>(*rhs);
    }

    if (lhs->is_offset() && rhs->is_offset()) {
        return branches_equal(
                lhs + static_cast<dimn_t>(*lhs), rhs + static_cast<dimn_t>(*rhs)
        );
    }

    return false;
}

struct print_walker {

    using pointer = const python::PyLieLetter*;
    std::stringstream ss;

    print_walker() : ss() {}

    void walk_single(pointer arg)
    {
        if (arg->is_offset()) {
            auto offset = static_cast<dimn_t>(*arg);
            walk_pair(arg + offset, arg + offset + 1);
        } else {
            ss << static_cast<let_t>(*arg);
        }
    }

    void walk_pair(pointer left, pointer right)
    {
        ss << '[';
        walk_single(left);
        ss << ',';
        walk_single(right);
        ss << ']';
    }

    string operator()(pointer tree)
    {
        walk_pair(tree, tree + 1);
        return ss.str();
    }
};

static typename python::PyLieKey::container_type trim_branch(
        const boost::container::small_vector_base<python::PyLieLetter>& tree,
        dimn_t start
)
{
    RPY_DBG_ASSERT(start == 0 || start == 1);
    if (tree.empty() || (tree.size() == 1 && start == 0)) { return {}; }
    if (tree.size() == 1 && start == 1) { return {tree[0]}; }
    if (tree.size() == 2) { return {tree[start]}; }

    if (!tree[start].is_offset()) { return {tree[start]}; }

    typename python::PyLieKey::container_type new_tree;
    new_tree.reserve(tree.size());
    dimn_t current = 0;
    dimn_t size = 0;

    auto visitor
            = [&new_tree, &current, &size](const python::PyLieLetter& node) {
                  ++current;
                  if (node.is_offset()) {
                      // Each offset points to a pair further down the tree
                      size += 2;
                      // point to the first
                      new_tree.emplace_back(
                              python::PyLieLetter::from_offset(size - current)
                      );
                  } else {
                      new_tree.emplace_back(node);
                  }
                  ++size;
              };

    walk_tree(tree.data() + start + static_cast<dimn_t>(tree[start]), visitor);

    new_tree.shrink_to_fit();
    return new_tree;
}

python::PyLieKey::PyLieKey(basis_type basis) : m_basis(std::move(basis)) {}
python::PyLieKey::PyLieKey(basis_type basis, let_t letter)
    : m_data{PyLieLetter::from_letter(letter)}, m_basis(std::move(basis))
{}
python::PyLieKey::PyLieKey(
        basis_type basis,
        const boost::container::small_vector_base<PyLieLetter>& data
)
    : m_data(data), m_basis(std::move(basis))
{}
python::PyLieKey::PyLieKey(basis_type basis, let_t left, let_t right)
    : m_data{PyLieLetter::from_letter(left), PyLieLetter::from_letter(right)},
      m_basis(std::move(basis))
{
    RPY_CHECK(left < right);
}
python::PyLieKey::PyLieKey(
        basis_type basis, let_t left, const python::PyLieKey& right
)
    : m_data{PyLieLetter::from_letter(left)}, m_basis(std::move(basis))
{
    RPY_CHECK(m_basis == right.m_basis);
    m_data.insert(m_data.end(), right.m_data.begin(), right.m_data.end());
    RPY_CHECK(!right.is_letter() || right.as_letter() > left);
}
python::PyLieKey::PyLieKey(
        basis_type basis, const python::PyLieKey& left,
        const python::PyLieKey& right
)
    : m_data{PyLieLetter::from_offset(2), PyLieLetter::from_offset(1 + left.degree())},
      m_basis(std::move(basis))
{
    m_data.insert(m_data.end(), left.m_data.begin(), left.m_data.end());
    m_data.insert(m_data.end(), right.m_data.begin(), right.m_data.end());
}

static python::PyLieKey::container_type
parse_key(const algebra::LieBasis& lbasis, key_type key)
{
    using namespace rpy::python;
    if (lbasis.letter(key)) {
        return {PyLieLetter::from_letter(lbasis.first_letter(key))};
    }

    auto keys = lbasis.parents(key);
    auto left_key = *keys.first;
    auto right_key = *keys.second;

    const bool left_letter = lbasis.letter(left_key);
    const bool right_letter = lbasis.letter(right_key);

    if (left_letter && right_letter) {
        return {PyLieLetter::from_letter(lbasis.first_letter(left_key)),
                PyLieLetter::from_letter(lbasis.first_letter(right_key))};
    }

    typename PyLieKey::container_type result;

    if (left_letter) {
        auto right_result = parse_key(lbasis, right_key);

        result.reserve(2 + right_result.size());
        result.push_back(PyLieLetter::from_letter(lbasis.first_letter(left_key))
        );
        result.push_back(PyLieLetter::from_offset(1));

        result.insert(result.cend(), right_result.begin(), right_result.end());
    } else if (right_letter) {
        auto left_result = parse_key(lbasis, left_key);

        result.reserve(2 + left_result.size());
        result.push_back(PyLieLetter::from_offset(2));
        result.push_back(PyLieLetter::from_letter(lbasis.first_letter(right_key)
        ));

        result.insert(result.cend(), left_result.begin(), left_result.end());
    } else {
        auto right_result = parse_key(lbasis, right_key);
        auto left_result = parse_key(lbasis, left_key);

        result.reserve(2 + left_result.size() + right_result.size());
        result.push_back(PyLieLetter::from_offset(2));
        result.push_back(PyLieLetter::from_offset(1 + left_result.size()));

        result.insert(result.cend(), left_result.begin(), left_result.end());
        result.insert(result.cend(), right_result.begin(), right_result.end());
    }

    return result;
}

python::PyLieKey::PyLieKey(const algebra::Context* ctx, key_type key)
    : m_data(parse_key(ctx->get_lie_basis(), key)),
      m_basis(ctx->get_lie_basis())
{}
python::PyLieKey::PyLieKey(algebra::LieBasis basis, key_type key)
    : m_data(parse_key(basis, key)), m_basis(std::move(basis))
{}
bool python::PyLieKey::is_letter() const noexcept { return m_data.size() == 1; }
let_t python::PyLieKey::as_letter() const
{
    RPY_CHECK(is_letter());
    return static_cast<let_t>(m_data[0]);
}
string python::PyLieKey::to_string() const
{
    if (m_data.size() == 1) {
        std::stringstream ss;
        ss << static_cast<let_t>(m_data[0]);
        return ss.str();
    }
    print_walker walker;
    return walker(m_data.data());
}
python::PyLieKey python::PyLieKey::lparent() const
{
    return PyLieKey(m_basis, trim_branch(m_data, 0));
}
python::PyLieKey python::PyLieKey::rparent() const
{
    return PyLieKey(m_basis, trim_branch(m_data, 1));
}
deg_t python::PyLieKey::degree() const
{
    return std::count_if(
            m_data.begin(), m_data.end(),
            [](const PyLieLetter& letter) { return !letter.is_offset(); }
    );
}
bool python::PyLieKey::equals(const python::PyLieKey& other) const noexcept
{
    const auto* lptr = m_data.data();
    const auto* rptr = other.m_data.data();
    return branches_equal(lptr, rptr) && branches_equal(++lptr, ++rptr);
}

namespace {

class ToLieKeyHelper
{
    using container_type = typename python::PyLieKey::container_type;
    RPY_UNUSED dimn_t size;
    RPY_UNUSED dimn_t current;
    algebra::LieBasis basis;
    deg_t width;
    deg_t depth = 0;
    let_t max_letter = 0;

public:
    explicit ToLieKeyHelper(algebra::LieBasis b, deg_t w)
        : basis(std::move(b)), size(2), current(0), width(w)
    {}

    container_type parse_list(const py::handle& obj)
    {
        if (py::len(obj) != 2) {
            RPY_THROW(
                    py::value_error,
                    "list items must contain exactly two elements"
            );
        }

        auto left = obj[py::int_(0)];
        auto right = obj[py::int_(1)];

        return parse_pair(left, right);
    }

    container_type parse_single(const py::handle& obj)
    {
        if (py::isinstance<py::list>(obj)) { return parse_list(obj); }
        if (py::isinstance<py::int_>(obj)) {
            auto as_let = obj.cast<let_t>();
            if (basis)
                if (as_let > max_letter) { max_letter = as_let; }
            return container_type{python::PyLieLetter::from_letter(as_let)};
        }
        RPY_THROW(py::type_error, "items must be either int or lists");
    }

    container_type parse_pair(const py::handle& left, const py::handle& right)
    {

        auto left_tree = parse_single(left);
        auto right_tree = parse_single(right);

        container_type result;
        dimn_t left_size;

        result.reserve(2 + left_tree.size() + right_tree.size());
        if (left_tree.size() == 1) {
            left_size = 0;
            result.push_back(left_tree[0]);
        } else {
            left_size = left_tree.size();
            result.push_back(python::PyLieLetter::from_offset(2));
        }
        if (right_tree.size() == 1) {
            result.push_back(right_tree[0]);
        } else {
            result.push_back(python::PyLieLetter::from_offset(1 + left_size));
        }
        if (left_tree.size() > 1) {
            result.insert(result.end(), left_tree.begin(), left_tree.end());
        }
        if (right_tree.size() > 1) {
            result.insert(result.end(), right_tree.begin(), right_tree.end());
        }

        return result;
    }

    container_type operator()(const py::handle& obj)
    {
        if (!py::isinstance<py::list>(obj)) {
            RPY_THROW(
                    py::type_error, "expected a list with exactly two elements"
            );
        }
        if (py::len(obj) != 2) {
            RPY_THROW(py::value_error, "expected list with exactly 2 elements");
        }
        return parse_pair(obj[py::int_(0)], obj[py::int_(1)]);
    }

    deg_t get_width()
    {
        if (width != 0 && max_letter > width) {
            RPY_THROW(py::value_error, "a letter exceeds the width");
        } else {
            width = max_letter;
        }
        return width;
    }

    algebra::LieBasis get_basis() const {
        RPY_CHECK(basis);

        return basis;
    }
};

}// namespace

static python::PyLieKey
make_lie_key(const py::args& args, const py::kwargs& kwargs)
{
    deg_t width = 0;
    deg_t depth = 0;

    if (args.empty()) {
        RPY_THROW(py::value_error, "argument cannot be empty");
    }
    algebra::LieBasis basis;
    if (kwargs.contains("basis")) {
        basis = kwargs["basis"].cast<algebra::LieBasis>();

        width = basis.width();
        depth = basis.depth();
    } else if (kwargs.contains("width") && kwargs.contains("depth")) {
        width = kwargs["width"].cast<deg_t>();
        depth = kwargs["depth"].cast<deg_t>();

        basis = algebra::get_context(
                width,
                depth,
                scalars::ScalarType::of<float>()
                )->get_lie_basis();
    } else {
        RPY_THROW(py::value_error,
                  "Either a basis or width/depth must be provided");
    }

    RPY_DBG_ASSERT(basis);


    if (py::isinstance<py::int_>(args[0])) {
        auto letter = args[0].cast<let_t>();
        if (letter > width) {
            RPY_THROW(py::value_error, "letter exceeds width");
        }
        return python::PyLieKey(std::move(basis), letter);
    }

    if (!py::isinstance<py::list>(args[0])) {
        RPY_THROW(py::type_error, "expected int or list");
    }

    ToLieKeyHelper helper(basis, width);

    return python::PyLieKey(helper.get_basis(), helper(args[0]));
}
void python::init_py_lie_key(py::module_& m)
{
    py::class_<PyLieKey> klass(m, "LieKey");
    klass.def(py::init(&make_lie_key));

    klass.def("__str__", &PyLieKey::to_string);
}

python::PyLieKey
python::parse_lie_key(const algebra::LieBasis& basis, const py::args& args, const py::kwargs&)
{
    auto len = py::len(args);
    if (args.empty() || len > 1) {
        RPY_THROW(py::value_error, "expected a letter or list");
    }

    if (py::isinstance<py::int_>(args[0])) {
        auto letter = args[0].cast<let_t>();
        return PyLieKey(basis, letter);
    }

    ToLieKeyHelper helper(basis, basis.width());
    return PyLieKey(basis, helper(args[0]));
}

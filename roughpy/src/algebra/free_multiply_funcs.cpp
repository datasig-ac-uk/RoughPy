//
// Created by user on 17/07/23.
//

#include "free_multiply_funcs.h"
#include <roughpy/algebra/context.h>

using namespace pybind11::literals;

using namespace rpy;

namespace {

static const char FREE_MULTIPLY_DOC[] = R"rpydoc()rpydoc";
py::object free_multiply(const py::object& left, const py::object& right)
{
    algebra::ConstRawUnspecifiedAlgebraType left_raw, right_raw;
    algebra::AlgebraType result_alg_type;
    algebra::context_pointer ctx;
    if (py::isinstance<algebra::FreeTensor>(left)) {
        const auto& left_real = left.cast<const algebra::FreeTensor&>();
        result_alg_type = algebra::AlgebraType::FreeTensor;
        ctx = left_real.context();
        left_raw = (&*left_real);
        if (py::isinstance<algebra::FreeTensor>(right)) {
            right_raw = &*right.cast<const algebra::FreeTensor&>();
        } else if (py::isinstance<algebra::ShuffleTensor>(right)) {
            right_raw = &*right.cast<const algebra::ShuffleTensor&>();
        } else {
            RPY_THROW(std::runtime_error,"argument 'right' must be of type "
                                     "FreeTensor or ShuffleTensor");
        }
    } else if (py::isinstance<algebra::ShuffleTensor>(left)) {
        result_alg_type = algebra::AlgebraType::ShuffleTensor;
        const auto& left_real = left.cast<const algebra::ShuffleTensor&>();
        ctx = left_real.context();
        left_raw = (&*left_real);
        if (py::isinstance<algebra::FreeTensor>(right)) {
            right_raw = &*right.cast<const algebra::FreeTensor&>();
        } else if (py::isinstance<algebra::ShuffleTensor>(right)) {
            right_raw = &*right.cast<const algebra::ShuffleTensor&>();
        } else {
            RPY_THROW(std::runtime_error,"argument 'right' must be of type "
                                     "FreeTensor or ShuffleTensor");
        }
    } else {
        RPY_THROW(std::runtime_error,"argument 'left' must be of type FreeTensor "
                                 "or ShuffleTensor");
    }

    RPY_CHECK(static_cast<bool>(ctx));

    auto result_ptr = ctx->free_multiply(left_raw, right_raw);

    switch (result_alg_type) {
        case algebra::AlgebraType::FreeTensor:
            return py::cast(algebra::FreeTensor(std::move(result_ptr)));
        case algebra::AlgebraType::ShuffleTensor:
            return py::cast(algebra::ShuffleTensor(std::move(result_ptr)));
        default: RPY_THROW(std::runtime_error, "unexpected result algebra type");
    }

    RPY_UNREACHABLE_RETURN(py::none());
}

static const char SHUFFLE_MULTIPLY_DOC[] = R"rpydoc()rpydoc";
py::object shuffle_multiply(const py::object& left, const py::object& right)
{
    algebra::ConstRawUnspecifiedAlgebraType left_raw, right_raw;
    algebra::AlgebraType result_alg_type;
    algebra::context_pointer ctx;
    if (py::isinstance<algebra::FreeTensor>(left)) {
        const auto& left_real = left.cast<const algebra::FreeTensor&>();
        result_alg_type = algebra::AlgebraType::FreeTensor;
        ctx = left_real.context();
        left_raw = (&*left_real);
        if (py::isinstance<algebra::FreeTensor>(right)) {
            right_raw = &*right.cast<const algebra::FreeTensor&>();
        } else if (py::isinstance<algebra::ShuffleTensor>(right)) {
            right_raw = &*right.cast<const algebra::ShuffleTensor&>();
        } else {
            RPY_THROW(std::runtime_error,"argument 'right' must be of type "
                                     "FreeTensor or ShuffleTensor");
        }
    } else if (py::isinstance<algebra::ShuffleTensor>(left)) {
        result_alg_type = algebra::AlgebraType::ShuffleTensor;
        const auto& left_real = left.cast<const algebra::ShuffleTensor&>();
        ctx = left_real.context();
        left_raw = (&*left_real);
        if (py::isinstance<algebra::FreeTensor>(right)) {
            right_raw = &*right.cast<const algebra::FreeTensor&>();
        } else if (py::isinstance<algebra::ShuffleTensor>(right)) {
            right_raw = &*right.cast<const algebra::ShuffleTensor&>();
        } else {
            RPY_THROW(std::runtime_error,"argument 'right' must be of type "
                                     "FreeTensor or ShuffleTensor");
        }
    } else {
        RPY_THROW(std::runtime_error,"argument 'left' must be of type FreeTensor "
                                 "or ShuffleTensor");
    }

    RPY_CHECK(static_cast<bool>(ctx));

    auto result_ptr = ctx->shuffle_multiply(left_raw, right_raw);

    switch (result_alg_type) {
        case algebra::AlgebraType::FreeTensor:
            return py::cast(algebra::FreeTensor(std::move(result_ptr)));
        case algebra::AlgebraType::ShuffleTensor:
            return py::cast(algebra::ShuffleTensor(std::move(result_ptr)));
        default: RPY_THROW(std::runtime_error, "unexpected result algebra type");
    }

    RPY_UNREACHABLE_RETURN(py::none());
}

static const char HALF_SHUFFLE_MULTIPLY_DOC[] = R"rpydoc()rpydoc";
py::object
half_shuffle_multiply(const py::object& left, const py::object& right)
{
    algebra::ConstRawUnspecifiedAlgebraType left_raw, right_raw;
    algebra::AlgebraType result_alg_type;
    algebra::context_pointer ctx;
    if (py::isinstance<algebra::FreeTensor>(left)) {
        const auto& left_real = left.cast<const algebra::FreeTensor&>();
        result_alg_type = algebra::AlgebraType::FreeTensor;
        ctx = left_real.context();
        left_raw = (&*left_real);
        if (py::isinstance<algebra::FreeTensor>(right)) {
            right_raw = &*right.cast<const algebra::FreeTensor&>();
        } else if (py::isinstance<algebra::ShuffleTensor>(right)) {
            right_raw = &*right.cast<const algebra::ShuffleTensor&>();
        } else {
            RPY_THROW(std::runtime_error,"argument 'right' must be of type "
                                     "FreeTensor or ShuffleTensor");
        }
    } else if (py::isinstance<algebra::ShuffleTensor>(left)) {
        result_alg_type = algebra::AlgebraType::ShuffleTensor;
        const auto& left_real = left.cast<const algebra::ShuffleTensor&>();
        ctx = left_real.context();
        left_raw = (&*left_real);
        if (py::isinstance<algebra::FreeTensor>(right)) {
            right_raw = &*right.cast<const algebra::FreeTensor&>();
        } else if (py::isinstance<algebra::ShuffleTensor>(right)) {
            right_raw = &*right.cast<const algebra::ShuffleTensor&>();
        } else {
            RPY_THROW(std::runtime_error,"argument 'right' must be of type "
                                     "FreeTensor or ShuffleTensor");
        }
    } else {
        RPY_THROW(std::runtime_error,"argument 'left' must be of type FreeTensor "
                                 "or ShuffleTensor");
    }

    RPY_CHECK(static_cast<bool>(ctx));

    auto result_ptr = ctx->half_shuffle_multiply(left_raw, right_raw);

    switch (result_alg_type) {
        case algebra::AlgebraType::FreeTensor:
            return py::cast(algebra::FreeTensor(std::move(result_ptr)));
        case algebra::AlgebraType::ShuffleTensor:
            return py::cast(algebra::ShuffleTensor(std::move(result_ptr)));
        default: RPY_THROW(std::runtime_error, "unexpected result algebra type");
    }

    RPY_UNREACHABLE_RETURN(py::none());
}

static const char ADJOINT_FREE_MULTIPLY_DOC[] = R"rpydoc()rpydoc";
py::object
adjoint_to_free_multiply(const py::object& multiplier, const py::object& arg)
{
    algebra::ConstRawUnspecifiedAlgebraType mul_raw, arg_raw;
    algebra::AlgebraType result_alg_type;
    algebra::context_pointer ctx;
    if (py::isinstance<algebra::FreeTensor>(multiplier)) {
        const auto& left_real = multiplier.cast<const algebra::FreeTensor&>();
        result_alg_type = algebra::AlgebraType::FreeTensor;
        ctx = left_real.context();
        mul_raw = (&*left_real);
        if (py::isinstance<algebra::FreeTensor>(arg)) {
            arg_raw = &*arg.cast<const algebra::FreeTensor&>();
        } else if (py::isinstance<algebra::ShuffleTensor>(arg)) {
            arg_raw = &*arg.cast<const algebra::ShuffleTensor&>();
        } else {
            RPY_THROW(std::runtime_error,"argument 'right' must be of type "
                                     "FreeTensor or ShuffleTensor");
        }
    } else if (py::isinstance<algebra::ShuffleTensor>(multiplier)) {
        result_alg_type = algebra::AlgebraType::ShuffleTensor;
        const auto& left_real = multiplier.cast<const
                algebra::ShuffleTensor&>();
        ctx = left_real.context();
        mul_raw = (&*left_real);
        if (py::isinstance<algebra::FreeTensor>(arg)) {
            arg_raw = &*arg.cast<const algebra::FreeTensor&>();
        } else if (py::isinstance<algebra::ShuffleTensor>(arg)) {
            arg_raw = &*arg.cast<const algebra::ShuffleTensor&>();
        } else {
            RPY_THROW(std::runtime_error,"argument 'right' must be of type "
                                     "FreeTensor or ShuffleTensor");
        }
    } else {
        RPY_THROW(std::runtime_error,"argument 'left' must be of type FreeTensor "
                                 "or ShuffleTensor");
    }

    RPY_CHECK(static_cast<bool>(ctx));

    auto result_ptr = ctx->adjoint_to_left_multiply_by(mul_raw, arg_raw);

    switch (result_alg_type) {
        case algebra::AlgebraType::FreeTensor:
            return py::cast(algebra::FreeTensor(std::move(result_ptr)));
        case algebra::AlgebraType::ShuffleTensor:
            return py::cast(algebra::ShuffleTensor(std::move(result_ptr)));
        default: RPY_THROW(std::runtime_error, "unexpected result algebra type");
    }

    RPY_UNREACHABLE_RETURN(py::none());
}

}// namespace

void rpy::python::init_free_multiply_funcs(py::module_& m)
{
    m.def("free_multiply", &free_multiply, FREE_MULTIPLY_DOC, "left"_a,
          "right"_a);
    m.def("shuffle_multiply", &shuffle_multiply, SHUFFLE_MULTIPLY_DOC, "left"_a,
          "right"_a);
    m.def("half_shuffle_multiply", &half_shuffle_multiply,
          HALF_SHUFFLE_MULTIPLY_DOC, "left"_a, "right"_a);
    m.def("adjoint_to_free_multiply", &adjoint_to_free_multiply,
          ADJOINT_FREE_MULTIPLY_DOC, "multiplier"_a, "arg"_a);
}

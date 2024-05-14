import re

HEADER_TEMPLATE = """
#ifndef ALGEBRA_HOST_KERNEL_{upper_name}_H
#define ALGEBRA_HOST_KERNEL_{upper_name}_H

#include "{header_name}.h"
#include <roughpy/device_support/host_kernel.h>
#include <roughpy/device_support/operators.h>

#include <roughpy/core/macros.h>

namespace rpy {{ namespace algebra {{

{extern_operators_block}

}}

namespace devices {{

{extern_kernels_block}

}}
}}

#endif //ALGEBRA_HOST_KERNEL_{upper_name}_H
"""

CPPFILE_TEMPLATE = """
#include "{snake_name}.h"

using namespace rpy;
using namespace rpy::algebra;

namespace rpy {{ namespace algebra {{

//template class {operator_name}<rpy::devices::operators::{operator}, {typename}>;

}}

namespace devices {{

template class HostKernel<algebra::{operator_name}<
    rpy::devices::operators::{operator}, {typename}>>;

}}
}}
"""

TYPES = [
    "float",
    "double"
]

OPERATORS = [
    ("VectorUnaryOperator", "Uminus"),
    ("VectorInplaceUnaryWithScalarOperator", "LeftScalarMultiply"),
    ("VectorInplaceUnaryWithScalarOperator", "RightScalarMultiply"),
    ("VectorInplaceUnaryWithScalarOperator", "RightScalarDivide"),
    ("VectorUnaryWithScalarOperator", "LeftScalarMultiply"),
    ("VectorUnaryWithScalarOperator", "RightScalarMultiply"),
    ("VectorUnaryWithScalarOperator", "RightScalarDivide"),
    ("VectorBinaryOperator", "Add"),
    ("VectorBinaryOperator", "Sub"),
    ("VectorInplaceBinaryOperator", "Add"),
    ("VectorInplaceBinaryOperator", "Sub"),
    ("VectorInplaceBinaryWithScalarOperator", "FusedLeftScalarMultiplyAdd"),
    ("VectorInplaceBinaryWithScalarOperator", "FusedRightScalarMultiplyAdd"),
    ("VectorInplaceBinaryWithScalarOperator", "FusedRightScalarDivideAdd"),
    ("VectorInplaceBinaryWithScalarOperator", "FusedLeftScalarMultiplySub"),
    ("VectorInplaceBinaryWithScalarOperator", "FusedRightScalarMultiplySub"),
    ("VectorInplaceBinaryWithScalarOperator", "FusedRightScalarDivideSub"),

]

EXTERN_TEMPLATE_OPERATOR = \
    "extern template class {operator_name}<" \
    "rpy::devices::operators::{operator}, {typename}>;"

EXTERN_TEMPLATE_KERNEL = \
    ("extern template class HostKernel<algebra::{operator_name}<"
     "rpy::devices::operators::{operator}, {typename}>>;")


def pascal_to_snake_case(name):
    return re.sub('(?!^)([A-Z]+)', r'_\1', name).lower()


def make_extern_kernel_block(operator_name, operator):
    kernel_block = [
        EXTERN_TEMPLATE_KERNEL.format(operator_name=operator_name,
                                      typename=typename, operator=operator)
        for typename in TYPES
    ]

    return "\n".join(kernel_block)


def make_extern_operators_block(operator_name, operator):
    operators_block = [
        # EXTERN_TEMPLATE_OPERATOR.format(operator_name=operator_name,
        #                                 typename=typename, operator=operator)
        # for typename in TYPES
    ]
    return "\n".join(operators_block)


def get_header(operator_name):
    return "vector_unary_operator" if "Unary" in operator_name \
        else "vector_binary_operator"


def make_headers(operator_name, operator):
    vec_name = "vector_inplace" if "Inplace" in operator_name else "vector"
    file = f"{vec_name}_{pascal_to_snake_case(operator)}.h"
    with open(file, "wt") as fp:
        fp.write(HEADER_TEMPLATE.format(upper_name=str.upper(operator),
                                        operator_type=operator_name,
                                        header_name=get_header(operator_name),
                                        extern_operators_block=make_extern_operators_block(
                                            operator_name, operator),
                                        extern_kernels_block=make_extern_kernel_block(
                                            operator_name, operator)))


def make_cpp(operator_name, operator, typename):
    vec_name = "vector_inplace" if "Inplace" in operator_name else "vector"
    file = f"{vec_name}_{pascal_to_snake_case(operator)}_{typename}.cpp"
    with open(file, "wt") as fp:
        fp.write(
            CPPFILE_TEMPLATE.format(
                snake_name=f"{vec_name}_{pascal_to_snake_case(operator)}",
                operator_name=operator_name,
                typename=typename, operator=operator))


def make_all():
    for (operator_name, operator) in OPERATORS:
        make_headers(operator_name, operator)
        for typename in TYPES:
            make_cpp(operator_name, operator, typename)


if __name__ == "__main__":
    make_all()

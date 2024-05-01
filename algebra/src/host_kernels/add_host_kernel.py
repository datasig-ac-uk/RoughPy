HEADER_TEMPLATE = """
#ifndef ALGEBRA_HOST_KERNEL_{upper_name}_H
#define ALGEBRA_HOST_KERNEL_{upper_name}_H

#include "{operator_type}.h"
#include <roughpy/device_support/host_kernel.h>

#include <roughpy/core/macros.h>

namespace ryp {{ namespace algebra {{

{extern_template_block}

}}}}

#endif //ALGEBRA_HOST_KERNEL_{upper_name}_H
"""

CPPFILE_TEMPLATE = """
#include "{lower_name}.h"

using namespace rpy;
using namespace rpy::algebra;

namespace rpy {{ namespace algebra {{

template class {operator_name}<{typename}, {operator}>;

template class HostKernel<{operator_name}<{typename}, {operator}>>;

}}}}
"""

TYPES = [
    "float",
    "double"
]

OPERATORS = [
    ("VectorUnaryOperator", "uminus"),
    ("VectorUnaryWithScalarOperator", "left_scalar_multiply"),
    ("VectorUnaryWithScalarOperator", "right_scalar_multiply"),
    ("VectorUnaryWithScalarOperator", "right_scalar_divide"),
    ("VectorBinaryOperator", "add"),
    ("VectorBinaryOperator", "sub"),
    ("VectorBinaryWithScalarOperator", "fused_left_scalar_multiply_add"),
    ("VectorBinaryWithScalarOperator", "fused_right_scalar_multiply_add"),
    ("VectorBinaryWithScalarOperator", "fused_scalar_divide_add"),
    ("VectorBinaryWithScalarOperator", "fused_left_scalar_multiply_sub"),
    ("VectorBinaryWithScalarOperator", "fused_right_scalar_multiply_sub"),
    ("VectorBinaryWithScalarOperator", "fused_scalar_divide_sub"),
]

EXTERN_TEMPLATE_OPERATOR = \
    """extern template class {operator_name}<{typename}, {operator}>;"""


def make_extern_block(operator_name, operator):
    return "\n".join([
        EXTERN_TEMPLATE_OPERATOR.format(operator_name=operator_name,
                                        typename=typename, operator=operator)
        for typename in TYPES
    ])


def make_headers(operator_name, operator):
    file = f"vector_{operator}.h"
    with open(file, "wt") as fp:
        fp.write(HEADER_TEMPLATE.format(upper_name=str.upper(operator),
                                        operator_type=operator_name,
                                        extern_template_block=make_extern_block(
                                            operator_name, operator)))


def make_cpp(operator_name, operator, typename):
    file = f"vector_{operator}_{typename}.cpp"
    with open(file, "wt") as fp:
        fp.write(
            CPPFILE_TEMPLATE.format(lower_name=f"vector_{operator.lower()}.h",
                                    operator_name=operator_name,
                                    typename=typename, operator=operator))


def make_all():
    for (operator_name, operator) in OPERATORS:
        make_headers(operator_name, operator)
        for typename in TYPES:
            make_cpp(operator_name, operator, typename)


if __name__ == "__main__":
    make_all()

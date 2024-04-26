//
// Created by sam on 4/26/24.
//

#include "generic_inplace_multiplication.h"

using namespace rpy;
using namespace rpy::algebra;
using namespace rpy::algebra::dtl;

void GenericInplaceMultiplication::eval_ss(
        VectorData& out,
        const VectorData& arg
) const
{}
void GenericInplaceMultiplication::eval_sd(
        VectorData& out,
        const VectorData& arg
) const
{}
void GenericInplaceMultiplication::eval_ds(
        VectorData& out,
        const VectorData& arg
) const
{}
void GenericInplaceMultiplication::eval_dd(
        VectorData& out,
        const VectorData& arg
) const
{}
void GenericInplaceMultiplication::operator()(
        VectorData& out,
        const VectorData& arg
) const
{
    switch (get_sparse_dense_config(out, arg)) {
        case 0b00: eval_ss(out, arg); break;
        case 0b01: eval_sd(out, arg); break;
        case 0b10: eval_ds(out, arg); break;
        case 0b11: eval_dd(out, arg); break;
        default: RPY_UNREACHABLE_RETURN();
    }
}

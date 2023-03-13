//
// Created by user on 03/03/23.
//

#include "algebra_base.h"

#include <ostream>

#include "context.h"

using namespace rpy::algebra;

void dtl::print_empty_algebra(std::ostream &os) {
    os << "{ }";
}

const rpy::scalars::ScalarType *dtl::context_to_scalars(context_pointer const &ptr) {
    return ptr->ctype();
}

UnspecifiedAlgebraType dtl::try_create_new_empty(context_pointer ctx, AlgebraType alg_type)  {
    return nullptr;
}

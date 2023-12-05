//
// Created by sam on 11/16/23.
//

#include "print.h"
#include "scalar_types.h"

namespace {

template <typename T>
void do_print(std::ostream& os, const T* val)
{
    os << *val;
}

}


void rpy::scalars::dtl::print_scalar_val(std::ostream& os,
    const void* ptr,
    const devices::TypeInfo& info)
{
#define X(TP) return do_print(os, (const TP*) ptr)
    DO_FOR_EACH_X(info)
#undef X
}

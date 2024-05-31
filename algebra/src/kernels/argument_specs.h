//
// Created by sam on 15/04/24.
//

#ifndef ARGUMENT_SPECS_H
#define ARGUMENT_SPECS_H

#include "common.h"

namespace rpy {
namespace algebra {
namespace dtl {

struct MutableVectorArg {
    using arg_type = VectorData&;

    template <typename D>
    using data = ArgData<VectorData, D>;
};
struct ConstVectorArg {
    using arg_type = const VectorData&;

    template <typename D>
    using data = ArgData<const VectorData, D>;
};
struct ConstScalarArg {
    using arg_type = scalars::ScalarCRef;

    template <typename D>
    using data = ArgData<scalars::ScalarCRef, D>;
};
struct MutableScalarArg {
    using arg_type = scalars::ScalarRef;

    template <typename D>
    using data = ArgData<scalars::ScalarRef, D>;
};

}// namespace dtl
}// namespace algebra
}// namespace rpy

#endif// ARGUMENT_SPECS_H

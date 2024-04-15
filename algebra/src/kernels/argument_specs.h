//
// Created by sam on 15/04/24.
//

#ifndef ARGUMENT_SPECS_H
#define ARGUMENT_SPECS_H


#include "common.h"



namespace rpy { namespace algebra { namespace dtl {


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
    using arg_type = const scalars::Scalar&;

    template <typename D>
    using data = ArgData<const scalars::Scalar, D>;
};
struct MutableScalarArg {
    using arg_type = scalars::Scalar&;

    template <typename D>
    using data = ArgData<scalars::Scalar, D>;
};


}}}

#endif //ARGUMENT_SPECS_H

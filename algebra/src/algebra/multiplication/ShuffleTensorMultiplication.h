//
// Created by sam on 19/09/24.
//

#ifndef SHUFFLETENSORMULTIPLICATION_H
#define SHUFFLETENSORMULTIPLICATION_H
#include "TensorMultiplication.h"

namespace rpy {
namespace algebra {

class ShuffleTensorMultiplication : TensorMultiplication
{

public:
    using TensorMultiplication::antipode;
    using TensorMultiplication::TensorMultiplication;

    void
    eval(scalars::ScalarVector& destination,
         const scalars::ScalarVector& left,
         const scalars::ScalarVector& right,
         const devices::operators::Operator& op,
         deg_t max_degree,
         deg_t min_left_degree,
         deg_t min_right_degree) const;

    void eval_inplace(
            scalars::ScalarVector& left,
            const scalars::ScalarVector& right,
            const devices::operators::Operator& op,
            deg_t max_degree,
            deg_t min_left_degree,
            deg_t min_right_degree
    ) const;
};

}// namespace algebra
}// namespace rpy

#endif// SHUFFLETENSORMULTIPLICATION_H

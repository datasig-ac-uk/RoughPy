//
// Created by sam on 19/09/24.
//

#include "ShuffleTensorMultiplication.h"

using namespace rpy;
using namespace rpy::algebra;

void ShuffleTensorMultiplication::eval(
        scalars::ScalarVector& destination,
        const scalars::ScalarVector& left,
        const scalars::ScalarVector& right,
        const devices::operators::Operator& op,
        deg_t max_degree,
        deg_t min_left_degree,
        deg_t min_right_degree
) const
{}
void ShuffleTensorMultiplication::eval_inplace(
        scalars::ScalarVector& left,
        const scalars::ScalarVector& right,
        const devices::operators::Operator& op,
        deg_t max_degree,
        deg_t min_left_degree,
        deg_t min_right_degree
) const
{}

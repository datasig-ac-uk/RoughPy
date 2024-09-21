//
// Created by sam on 19/09/24.
//

#include "FreeTensorMultiplication.h"

using namespace rpy;
using namespace rpy::algebra;

devices::KernelLaunchParams
FreeTensorMultiplication::get_launch_params(const devices::KernelArguments& args
) const noexcept
{}
void FreeTensorMultiplication::eval(
        scalars::ScalarVector& destination,
        const scalars::ScalarVector& left,
        const scalars::ScalarVector& right,
        const devices::operators::Operator& op,
        deg_t max_degree,
        deg_t min_left_degree,
        deg_t min_right_degree
) const
{}
void FreeTensorMultiplication::eval_inplace(
        scalars::ScalarVector& left,
        const scalars::ScalarVector& right,
        const devices::operators::Operator& op,
        deg_t max_degree,
        deg_t min_left_degree,
        deg_t min_right_degree
) const
{}

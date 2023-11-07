//
// Created by sam on 12/05/23.
//

#include "random_impl.h"

const char* rpy::scalars::dtl::rng_type_getter<std::mt19937_64>::name
        = "mt19937";
const char* rpy::scalars::dtl::rng_type_getter<pcg64>::name = "pcg";

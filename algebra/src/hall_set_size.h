//
// Created by user on 06/03/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_HALL_SET_SIZE_H
#define ROUGHPY_ALGEBRA_SRC_HALL_SET_SIZE_H

#include <roughpy/core/implementation_types.h>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>


namespace rpy { namespace algebra {


class HallSetSizeHelper {
    deg_t m_width;
    deg_t m_depth;

    /*
     * The first 50 values of the Mobius function
     * should be plenty, but we should include
     * the ability to compute more if we need to.
     */
    std::vector<int32_t> m_mobius = {
         0, // included so the index = argument
         1, -1, -1,  0, -1,  1, -1,  0,  0,  1,
        -1,  0, -1,  1,  1,  0, -1,  0, -1,  0,
         1,  1, -1,  0,  0,  1,  0,  0, -1, -1,
        -1,  0,  1,  1,  1,  0, -1,  1,  1,  0,
        -1, -1, -1,  0,  0,  1, -1,  0,  0,  0
    };

public:

    HallSetSizeHelper(deg_t width, deg_t depth);


    dimn_t operator()(int k);


};




}}


#endif//ROUGHPY_ALGEBRA_SRC_HALL_SET_SIZE_H

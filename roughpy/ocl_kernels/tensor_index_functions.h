//
// Created by sam on 07/09/23.
//

#ifndef ROUGHPY_TENSOR_INDEX_FUNCTIONS_H
#define ROUGHPY_TENSOR_INDEX_FUNCTIONS_H

#include "kernel_types.h"


inline size_t
split_index(const size_t at, const size_t left, RPY_ADDR_PRIVT size_t* right)
{
    size_t div = left / at;
    *right = left - div * at;
    return div;
}


inline size_t reverse_index(const int degree, const int width, size_t index) {
    size_t result = 0;
    size_t rem;
    for (int i=0; i<degree; ++i) {
        result *= width;
        index = split_index(width, index, &rem);
        result += rem;
    }
    return result;
}


#endif// ROUGHPY_TENSOR_INDEX_FUNCTIONS_H

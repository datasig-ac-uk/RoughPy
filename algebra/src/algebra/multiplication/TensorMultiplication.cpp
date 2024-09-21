//
// Created by sam on 19/09/24.
//

#include "TensorMultiplication.h"

using namespace rpy;
using namespace rpy::algebra;
TensorMultiplication::TensorMultiplication(
        deg_t width,
        deg_t depth,
        deg_t tile_letters
)
    : m_width(width), m_depth(depth), m_tile_letters(tile_letters)
{}

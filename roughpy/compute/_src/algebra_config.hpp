#ifndef ROUGHPY_COMPUTE__SRC_ALGEBRA_CONFIG_HPP
#define ROUGHPY_COMPUTE__SRC_ALGEBRA_CONFIG_HPP

#include <cstdint>

namespace rpy::compute {

struct AlgebraConfig {
    int32_t width;
    int32_t depth;
    int32_t lhs_max_degree=-1;
    int32_t rhs_max_degree=-1;
    int32_t lhs_min_degree = 0;
    int32_t rhs_min_degree = 0;
    void const* basis_data = nullptr;
};


} // namespace rpy::compute


#endif //ROUGHPY_COMPUTE__SRC_ALGEBRA_CONFIG_HPP

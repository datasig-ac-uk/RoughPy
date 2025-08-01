#ifndef ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_ANTIPODE_HPP
#define ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_ANTIPODE_HPP


#include <algorithm>

#include "roughpy_compute/common/cache_array.hpp"
#include "roughpy_compute/dense/views.hpp"

namespace rpy::compute::basic {
inline namespace v1 {


struct BasicAntipodeConfig {
    static constexpr int32_t tile_letters = 1;
    static constexpr int32_t max_width = 5;

};


template <typename S, typename AntipodeConfig>
void ft_antipode(DenseTensorView<S*> out, DenseTensorView<S const*> arg, AntipodeConfig const& config)
{
    using Degree = typename DenseTensorView<S*>::Degree;
    using Index = typename DenseTensorView<S*>::Index;


    CacheArray


}


} // version namespace
} // namespace rpy::compute::basic


#endif // ROUGHPY_COMPUTE_DENSE_BASIC_FREE_TENSOR_ANTIPODE_HPP
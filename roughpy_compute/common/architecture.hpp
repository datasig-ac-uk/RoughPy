#ifndef ROUGHPY_COMPUTE_COMMON_ARCHITECTURE_HPP
#define ROUGHPY_COMPUTE_COMMON_ARCHITECTURE_HPP

#include <cstdint>

namespace rpy::compute {


struct NativeArchitecture
{
    using Size = std::size_t;
    using Index = std::ptrdiff_t;

    using Degree = std::int32_t;
};

} // namespace rpy::compute

#endif //ROUGHPY_COMPUTE_COMMON_ARCHITECTURE_HPP

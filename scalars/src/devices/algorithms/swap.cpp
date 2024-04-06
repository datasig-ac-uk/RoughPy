
#include "algorithm_impl.h"
#include "devices/algorithms.h"
#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;

namespace {

template <typename S, typename T>
struct SwapFunctor {
    void operator()(Buffer& left, Buffer& right) const
    {
        RPY_THROW(
                std::runtime_error,
                "cannot swap buffers containing different types"
        );
    }
};

template <typename T>
struct SwapFunctor<T, T> {
    void operator()(Buffer& left, Buffer& right) const
    {
        auto lslice = left.as_mut_slice<T>();
        auto rslice = right.as_mut_slice<T>();
        rpy::ranges::swap(lslice, rslice);;
    }
};

}// namespace

void AlgorithmDrivers::swap_ranges(Buffer& left, Buffer& right) const
{
    auto left_host_mapped = left.map();
    auto right_host_mapped = right.map();
    algorithms::do_algorithm<void, SwapFunctor>(left_host_mapped, right_host_mapped);
}

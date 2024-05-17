

#include "algorithm_impl.h"
#include "devices/algorithms.h"

#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;

namespace {

template <typename S, typename T>
struct LexicographicalCompareFunctor {
    bool operator()(const Buffer& left, const Buffer& right) const
    {
        return rpy::ranges::lexicographical_compare(
                left.as_slice<S>(),
                right.as_slice<T>()
        );
    }
};

}// namespace


bool AlgorithmDrivers::lexicographical_compare(
        const Buffer& left,
        const Buffer& right
) const
{
    const auto left_host_view = left.map();
    const auto right_host_view = right.map();
    return algorithms::do_algorithm<bool, LexicographicalCompareFunctor>(
            left_host_view,
            right_host_view
    );
}

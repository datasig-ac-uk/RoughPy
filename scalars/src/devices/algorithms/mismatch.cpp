
#include "algorithm_impl.h"
#include "devices/algorithms.h"
#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;

namespace {

template <typename S, typename T>
struct MismatchFunctor {
    optional<dimn_t> operator()(const Buffer& left, const Buffer& right) const
    {
        const auto lslice = left.as_slice<S>();
        const auto rslice = right.as_slice<T>();

        const auto lbegin = lslice.begin();
        const auto lend = lslice.end();
        const auto rbegin = rslice.begin();
        const auto rend = rslice.end();

        auto result = rpy::ranges::mismatch(lbegin, lend, rbegin, rend);

        if (result.in1 != lend) {
            return static_cast<dimn_t>(result.in1 - lbegin);
        }
        if (result.in2 != rend) {
            return static_cast<dimn_t>(result.in2 - rbegin);
        }

        return {};
    }
};
}// namespace

optional<dimn_t>
AlgorithmDrivers::mismatch(const Buffer& left, const Buffer& right) const
{
    const auto left_host_view = left.map();
    const auto right_host_view = right.map();
    return algorithms::do_algorithm<optional<dimn_t>, MismatchFunctor>(
        left_host_view, right_host_view
    );
}

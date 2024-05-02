#include "algorithm_impl.h"
#include "devices/algorithms.h"

#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;

namespace {

template <typename T>
struct UpperBoundFunctor {
    optional<dimn_t> operator()(const Buffer& buffer, ConstReference value) const
    {
        auto slice = buffer.as_slice<T>();
        auto begin = slice.begin();
        auto end = slice.end();

        auto pos = rpy::ranges::upper_bound(begin, end, value.value<T>());
        if (pos != end) { return static_cast<dimn_t>(pos - begin); }
        return {};
    }
};

}// namespace

optional<dimn_t>
AlgorithmDrivers::upper_bound(const Buffer& buffer, ConstReference value) const
{
    const auto host_view = buffer.map();
    return algorithms::do_algorithm<optional<dimn_t>, UpperBoundFunctor>(
            host_view,
            value
    );
}
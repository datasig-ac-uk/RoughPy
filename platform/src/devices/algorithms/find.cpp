//
// Created by sam on 4/6/24.
//

#include "algorithm_impl.h"
#include "devices/algorithms.h"

#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;
namespace {

template <typename T>
struct FindFunctor {
    optional<dimn_t> operator()(const Buffer& buffer, ConstReference value) const
    {
        const auto slice = buffer.as_slice<T>();
        const auto begin = slice.begin();
        const auto end = slice.end();

        auto found = rpy::ranges::find(begin, end, value.value<T>());

        if (found != end) { return static_cast<dimn_t>(found - begin); }
        return {};
    }
};
};// namespace

optional<dimn_t>
AlgorithmDrivers::find(const Buffer& buffer, ConstReference value) const
{
    const auto host_view = buffer.map();
    return algorithms::do_algorithm<optional<dimn_t>, FindFunctor>(
            host_view,
            value
    );
}

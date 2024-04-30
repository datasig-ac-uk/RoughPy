

#include "algorithm_impl.h"
#include "devices/algorithms.h"

#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;

namespace {

template <typename T>
struct CountFunctor {
    RPY_NO_DISCARD dimn_t
    operator()(const Buffer& buffer, ConstReference value) const
    {
        auto slice = buffer.as_slice<T>();
        return rpy::ranges::count(slice, value.value<T>());
    }
};

}// namespace

dimn_t AlgorithmDrivers::count(const Buffer& buffer, ConstReference value) const
{
    const auto host_view = buffer.map();
    return algorithms::do_algorithm<dimn_t, CountFunctor>(host_view, value);
}

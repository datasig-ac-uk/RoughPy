#include "algorithm_impl.h"
#include "devices/algorithms.h"

#include <roughpy/core/ranges.h>

using namespace rpy;
using namespace rpy::devices;

namespace {

template <typename T>
struct MaxFunctor {
    void operator()(const Buffer& buffer, Reference out) const
    {
        out.value<T>() = rpy::ranges::max(buffer.as_slice<T>());
    }
};

}// namespace

void AlgorithmDrivers::max(const Buffer& buffer, Reference out) const
{
    const auto host_view = buffer.map();
    algorithms::do_algorithm<void, MaxFunctor>(host_view, std::move(out));
}

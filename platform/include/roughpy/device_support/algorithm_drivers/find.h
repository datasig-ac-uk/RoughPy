#ifndef ROUGHPY_DEVICE_SUPPORT_FIND_H_
#define ROUGHPY_DEVICE_SUPPORT_FIND_H_

#include "algorithm_base.h"
#include <roughpy/core/ranges.h>
#include <roughpy/core/traits.h>

namespace rpy {
namespace devices {

namespace __find {

template <typename S, typename T, bool Cond = true>
struct Func : dtl::AlgorithmFunctionBase<S, T, Cond> {
    optional<dimn_t>
    operator()(const Buffer& buffer, ConstReference value) const
    {
        const auto view = buffer.map();
        const auto& find_val = value.value<T>();
        const auto as_slice = view.as_slice<S>();

        const auto begin = as_slice.begin();
        const auto end = as_slice.end();

        auto found = rpy::ranges::find(begin, end, find_val);

        if (found != end) { return ranges::distance(begin, found); }
        return {};
    }
};

template <typename S, typename T>
struct Func<S, T, false> : dtl::AlgorithmFunctionBase<S, T, false> {
    RPY_NO_RETURN optional<dimn_t>
    operator()(const Buffer&, ConstReference) const
    {
        RPY_THROW(
                std::runtime_error,
                string_join(
                        ' ',
                        "types",
                        type_id_of<S>,
                        "and",
                        type_id_of<T>,
                        "are not convertible"
                )
        );
    }
};

template <typename S, typename T>
using func = Func<S, T, is_equal_comparable_v<S, T>>;

}// namespace __find

template <typename S, typename T>
constexpr __find::func<S, T> find_func;

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_FIND_H_

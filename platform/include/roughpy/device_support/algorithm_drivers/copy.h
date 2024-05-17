#ifndef ROUGHPY_DEVICE_SUPPORT_COPY_H_
#define ROUGHPY_DEVICE_SUPPORT_COPY_H_

#include <roughpy/core/ranges.h>
#include <roughpy/core/traits.h>

#include "algorithm_base.h"

namespace rpy {
namespace devices {

namespace __copy {

template <typename S, typename T, bool Cond>
struct Func : dtl::AlgorithmFunctionBase<S, T, Cond> {
    void
    operator()(Buffer& destination_buffer, const Buffer& source_buffer) const
    {
        const auto source_view = source_buffer.map();
        auto destination_view = destination_buffer.map();
        const auto source_slice = source_view.as_slice<S>();
        auto destination_slice = destination_view.as_slice<S>();

        std::copy(
                source_slice.begin(),
                source_slice.end(),
                destination_slice.begin()
        );
    }
};

template <typename S, typename T>
struct Func<S, T, false> : dtl::AlgorithmFunctionBase<S, T, false> {
    RPY_NO_RETURN void operator()(Buffer&, const Buffer&) const
    {
        RPY_THROW(
                std::runtime_error,
                string_join(
                        ' ',
                        "type",
                        type_id_of<S>(),
                        "is not constructible from",
                        type_id_of<T>()
                )
        );
    }
};

template <typename S, typename T>
using func = Func<S, T, is_constructible_v<S, const T&>>;

}// namespace __copy

template <typename S, typename T>
constexpr __copy::func<S, T> copy_func{};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_COPY_H_

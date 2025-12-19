#ifndef ROUGHPY_JAX_SRC_CPU_BATCHING_LOOP_HPP
#define ROUGHPY_JAX_SRC_CPU_BATCHING_LOOP_HPP

#include <array>
#include <algorithm>
#include <functional>
#include <utility>

#include "xla_includes.hpp"
#include "xla_common.hpp"



namespace rpy::jax::cpu {

namespace detail {

template <typename Head, typename... Tail>
ffi::Error batch_config_recurse(
        size_t ncore_dims,
        ffi::Span<const int64_t> batch_dims,
        Head & head,
        Tail &... tail
)
{
    const ffi::Span<const int64_t> dims = shape(head);
    const auto these_batch_dims = dims.first(dims.size() - ncore_dims);

    for (size_t i=0; i<these_batch_dims.size(); ++i) {
        if (these_batch_dims[i] != batch_dims[i]) {
            return ffi::Error::InvalidArgument("mismatched batch dimensions");
        }
    }

    if constexpr (sizeof...(Tail) > 0) {
        return batch_config_recurse(ncore_dims, batch_dims, tail...);
    }

    return ffi::Error::Success();
}

template <typename Head, typename... Tail>
ffi::ErrorOr<int64_t> get_batch_dims(size_t ncore_dims, Head& head, Tail&... tail)
{
    const ffi::Span<const int64_t> dims = shape(head);
    auto batch_dims = dims.first(dims.size() - ncore_dims);

    if constexpr (sizeof...(Tail) > 0) {
        auto status = batch_config_recurse(ncore_dims, batch_dims, tail...);
        if (!status.success()) {
            return ffi::Unexpected(status);
        }
    }

    return std::accumulate(batch_dims.begin(), batch_dims.end(), int64_t{1}, std::multiplies<>{});
}

struct BaseAccessor
{
    int64_t stride;

    template <typename Array>
    BaseAccessor(Array& stride_array, size_t ncore_dims) noexcept
    {
        auto core_dims = shape(stride_array).last(ncore_dims);
        stride = std::accumulate(core_dims.begin(), core_dims.end(), int64_t{1}, std::multiplies<>{});
    }
};

template <typename T, typename Array>
struct ArrayAccessor;

template <typename T>
struct ArrayAccessor<T, ffi::AnyBuffer> : BaseAccessor
{
    ffi::AnyBuffer array_;

    ArrayAccessor(ffi::AnyBuffer array, size_t ncore_dims) noexcept
        : BaseAccessor(array, ncore_dims), array_(std::move(array))
    {}

    T const* operator()(const int64_t batch_idx) noexcept
    {
        return array_.typed_data<T>() + stride * batch_idx;
    }
};

template <typename T>
struct ArrayAccessor<T, ffi::Result<ffi::AnyBuffer>> : BaseAccessor
{
    ffi::Result<ffi::AnyBuffer> array_;

    ArrayAccessor(ffi::Result<ffi::AnyBuffer> array, size_t ncore_dims) noexcept
        : BaseAccessor(array, ncore_dims), array_(std::move(array))
    {}

    T* operator()(const int64_t batch_idx) noexcept
    {
        return array_->typed_data<T>() + stride * batch_idx;
    }
};


template <typename T, typename Array>
ArrayAccessor<T, Array> make_accessor(Array array, size_t ncore_dims) noexcept
{
    using BufT = std::remove_const_t<std::remove_cv_t<Array>>;
    return ArrayAccessor<T, BufT>(std::move(array), ncore_dims);
}

template <typename Fn, typename... Accessors>
ffi::Error do_batch_loop(int64_t batch_size, Fn&& fn, Accessors&&... accessors) noexcept
{
    for (int64_t i = 0; i < batch_size; ++i) {
        RPY_XLA_SUCCESS_OR_RETURN(fn(accessors(i)...));
    }

    return ffi::Error::Success();
}


}


template <typename Fn, typename... ArrayArgs>
ffi::Error batching_loop(
    Fn&& function,
    ArrayArgs&&... arrays
    )
{
    using T = typename Fn::Scalar;
    auto result = detail::get_batch_dims(Fn::core_dims, arrays...);
    if (result.has_error()) {
        return result.error(); // take the error
    }

    const auto n_batches = result.value();

    return detail::do_batch_loop(
            n_batches,
            std::forward<Fn>(function),
            detail::make_accessor<T>(std::forward<ArrayArgs>(arrays), Fn::core_dims)...
    );
}



template <template <ffi::DataType> class Fn, typename StaticArgs, typename... Args>
ffi::Error select_implementation_and_go(StaticArgs&& static_args, ffi::DataType dtype, Args&&... args) noexcept
{
    switch (dtype) {
        case ffi::DataType::F64:
            return batching_loop(
                    Fn<ffi::DataType::F64>{
                            std::forward<StaticArgs>(static_args)
                    },
                    std::forward<Args>(args)...
            );
        case ffi::DataType::F32:
            return batching_loop(
                    Fn<ffi::DataType::F32>{
                            std::forward<StaticArgs>(static_args)
                    },
                    std::forward<Args>(args)...
            );
        default:
            return ffi::Error::InvalidArgument("unsupported data type");
    }
}



}

#endif// ROUGHPY_JAX_SRC_CPU_BATCHING_LOOP_HPP

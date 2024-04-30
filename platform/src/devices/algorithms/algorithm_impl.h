//
// Created by sam on 4/3/24.
//

#ifndef ALGORITHM_IMPL_H
#define ALGORITHM_IMPL_H

#include <roughpy/core/macros.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include "devices/core.h"
#include "devices/device_handle.h"
#include "devices/type.h"

namespace rpy {
namespace devices {
namespace algorithms {

namespace dtl {
RPY_NO_RETURN RPY_LOCAL inline void report_algo_error(
        const Type* const& type,
        const Device& device,
        string_view algo_name
)
{
    string device_name;
    switch (device->type()) {
        case CPU: device_name = "CPU"; break;
        case CUDA: device_name = "CUDA"; break;
        case CUDAHost: device_name = "CUDAHost"; break;
        case OpenCL: device_name = "OpenCL"; break;
        case Vulkan: device_name = "Vulkan"; break;
        case Metal: device_name = "Metal"; break;
        case VPI: device_name = "VPI"; break;
        case ROCM: device_name = "ROCm"; break;
        case ROCMHost: device_name = "ROCmHost"; break;
        case ExtDev: device_name = "ExtDev"; break;
        case CUDAManaged: device_name = "CUDAManaged"; break;
        case OneAPI: device_name = "OneAPI"; break;
        case WebGPU: device_name = "WebGPU"; break;
        case Hexagon: device_name = "Hexagon"; break;
    }

    RPY_THROW(
            std::runtime_error,
            "no supported implementation of \"" + string(algo_name)
                    + "\" for type \"" + string(type->name())
                    + "\" on device \"" + device_name + '"'
    );
}

template <typename F, typename... Args>
auto apply_algo_function(F&& function, Args&&... args)
        -> decltype(function(std::forward<Args>(args)...))
{
    return function(std::forward<Args>(args)...);
}

template <typename T>
struct WrappedBufferRef {
    using type = remove_const_t<T>;

    using value_type = conditional_t<is_const_v<T>, const Buffer, Buffer>;
    using pointer_type = value_type*;
    using ref_type = value_type&;
    pointer_type p_ref;

    WrappedBufferRef(ref_type buffer) : p_ref(&buffer) {}
    operator ref_type() noexcept { return *p_ref; }

    constexpr ref_type ref() noexcept { return *p_ref; }
};

#define DO_FOR_EACH_TP(info)                                                   \
    switch (info.code) {                                                       \
        case TypeCode::Int:                                                    \
            switch (info.bytes) {                                              \
                case 1: X(int8_t);                                             \
                case 2: X(int16_t);                                            \
                case 4: X(int32_t);                                            \
                case 8: X(int64_t);                                            \
            }                                                                  \
            break;                                                             \
        case TypeCode::UInt:                                                   \
            switch (info.bytes) {                                              \
                case 1: X(uint8_t);                                            \
                case 2: X(uint16_t);                                           \
                case 4: X(uint32_t);                                           \
                case 8: X(uint64_t);                                           \
            }                                                                  \
            break;                                                             \
        case TypeCode::Float:                                                  \
            switch (info.bytes) {                                              \
                case 4: X(float);                                              \
                case 8: X(double);                                             \
            }                                                                  \
            break;                                                             \
        default: break;                                                        \
    }

template <typename S, typename T>
using copy_const_t = conditional_t<is_const_v<S>, const T, T>;

template <typename Result, template <typename...> class Functor, typename... Ts>
class AlgoFunctorEvalutor
{
    template <typename... Rs, typename F = Functor<Rs...>>
    static constexpr true_type test(void*);

    template <typename...>
    static constexpr false_type test(...);

    using marker_t = decltype(test<Ts...>(nullptr));

    template <typename T>
    using next_t
            = AlgoFunctorEvalutor<Result, Functor, Ts..., WrappedBufferRef<T>>;

    template <typename... Args>
    static Result eval_impl(
            const std::true_type& RPY_UNUSED_VAR marker,
            Ts... parsed,
            Args&&... args
    )
    {
        Functor<typename Ts::type...> func;
        return func(parsed.ref()..., std::forward<Args>(args)...);
    }

    template <typename Head, typename... Args>
    static Result eval_impl(
            const std::false_type& RPY_UNUSED_VAR marker,
            Ts... parsed,
            Head& head,
            Args&&... args
    )
    {
        static_assert(
                is_same_v<decay_t<Head>, Buffer>,
                "Head must be a Buffer or or const Buffer"
        );

        const auto* type = head.content_type();
        const auto info = type->type_info();

#define X(tp)                                                                  \
    return next_t<copy_const_t<Head, tp>>::eval(                               \
            parsed...,                                                         \
            WrappedBufferRef<copy_const_t<Head, tp>>(head),                    \
            std::forward<Args>(args)...                                        \
    )
        DO_FOR_EACH_TP(info)
#undef X

        RPY_THROW(
                std::runtime_error,
                "unsupported type " + string(type->name())
        );
    }

public:
    template <typename... Args>
    static Result eval(const Ts&... parsed, Args&&... args)
    {
        return eval_impl(marker_t(), parsed..., std::forward<Args>(args)...);
    }
};
}// namespace dtl

template <
        typename Result,
        template <typename...>
        class Functor,
        typename... Args>
Result do_algorithm(Args&&... args)
{
    return dtl::AlgoFunctorEvalutor<Result, Functor>::eval(
            std::forward<Args>(args)...
    );
}

}// namespace algorithms
}// namespace devices
}// namespace rpy

#endif// ALGORITHM_IMPL_H

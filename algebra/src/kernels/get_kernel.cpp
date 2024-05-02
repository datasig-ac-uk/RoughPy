//
// Created by sam on 4/17/24.
//

#include "kernels/kernel.h"
#include "roughpy/devices/host_device.h"

#include <roughpy/core/types.h>

#include "host_kernels/vector_add.h"
#include "host_kernels/vector_inplace_add.h"
#include "host_kernels/vector_inplace_fused_left_scalar_multiply_add.h"
#include "host_kernels/vector_inplace_fused_left_scalar_multiply_sub.h"
#include "host_kernels/vector_inplace_fused_right_scalar_divide_add.h"
#include "host_kernels/vector_inplace_fused_right_scalar_divide_sub.h"
#include "host_kernels/vector_inplace_fused_right_scalar_multiply_add.h"
#include "host_kernels/vector_inplace_fused_right_scalar_multiply_sub.h"
#include "host_kernels/vector_inplace_left_scalar_multiply.h"
#include "host_kernels/vector_inplace_right_scalar_multiply.h"
#include "host_kernels/vector_inplace_sub.h"
#include "host_kernels/vector_left_scalar_multiply.h"
#include "host_kernels/vector_right_scalar_divide.h"
#include "host_kernels/vector_right_scalar_multiply.h"
#include "host_kernels/vector_sub.h"
#include "host_kernels/vector_uminus.h"

using namespace rpy;
using namespace rpy::devices;

namespace {
template <typename T>
struct IdOfType;

template <>
struct IdOfType<float> {
    static constexpr string_view value = "f32";
};

template <>
struct IdOfType<double> {
    static constexpr string_view value = "f64";
};

template <typename O>
struct SuffixOfType;

template <template <typename...> class Op, typename T>
struct SuffixOfType<algebra::VectorUnaryOperator<Op, T>> {
    static constexpr string_view value = "Dd";
};

template <template <typename...> class Op, typename T>
struct SuffixOfType<algebra::VectorUnaryWithScalarOperator<Op, T>> {
    static constexpr string_view value = "Ddv";
};

template <template <typename...> class Op, typename T>
struct SuffixOfType<algebra::VectorInplaceUnaryOperator<Op, T>> {
    static constexpr string_view value = "D";
};

template <template <typename...> class Op, typename T>
struct SuffixOfType<algebra::VectorInplaceUnaryWithScalarOperator<Op, T>> {
    static constexpr string_view value = "Dv";
};

template <template <typename...> class Op, typename T>
struct SuffixOfType<algebra::VectorBinaryOperator<Op, T>> {
    static constexpr string_view value = "Ddd";
};

template <template <typename...> class Op, typename T>
struct SuffixOfType<algebra::VectorInplaceBinaryOperator<Op, T>> {
    static constexpr string_view value = "Dd";
};

template <template <typename...> class Op, typename T>
struct SuffixOfType<algebra::VectorBinaryWithScalarOperator<Op, T>> {
    static constexpr string_view value = "Dddv";
};

template <template <typename...> class Op, typename T>
struct SuffixOfType<algebra::VectorInplaceBinaryWithScalarOperator<Op, T>> {
    static constexpr string_view value = "Ddv";
};

class RegisterHostKernels
{

    template <template <typename...> class Op, typename T>
    using VUO = algebra::VectorUnaryOperator<Op, T>;

    template <template <typename...> class Op, typename T>
    using VIUO = algebra::VectorInplaceUnaryOperator<Op, T>;

    template <template <typename...> class Op, typename T>
    using VUWSO = algebra::VectorUnaryWithScalarOperator<Op, T>;

    template <template <typename...> class Op, typename T>
    using VIUWSO = algebra::VectorInplaceUnaryWithScalarOperator<Op, T>;

    template <template <typename...> class Op, typename T>
    using VBO = algebra::VectorBinaryOperator<Op, T>;

    template <template <typename...> class Op, typename T>
    using VBWSO = algebra::VectorBinaryWithScalarOperator<Op, T>;

    template <template <typename...> class Op, typename T>
    using VIBO = algebra::VectorInplaceBinaryOperator<Op, T>;

    template <template <typename...> class Op, typename T>
    using VIBWSO = algebra::VectorInplaceBinaryWithScalarOperator<Op, T>;

    template <
            template <template <typename...> class, typename...>
            class Wrapper,
            template <typename...>
            class Op,
            typename T>
    Kernel make_kernel(const string& name)
    {
        return Kernel(new HostKernel<Wrapper<Op, T>>(
                name + string(SuffixOfType<Wrapper<Op, T>>::value)
        ));
    }

    template <typename T>
    void register_all_kernels(const HostDevice& host)
    {
        using namespace operators;

        static const auto type_id = static_cast<string>(IdOfType<T>::value);

        host->register_kernel(make_kernel<VUO, Uminus, T>("uminus_" + type_id));
        host->register_kernel(make_kernel<VUWSO, LeftScalarMultiply, T>(
                "left_scalar_multiply_" + type_id + "_"
        ));
        host->register_kernel(make_kernel<VUWSO, RightScalarMultiply, T>(
                "right_scalar_mutliply_" + type_id + "_"
        ));

        host->register_kernel(make_kernel<VIUWSO, LeftScalarMultiply, T>(
                "inplace_left_scalar_multiply_" + type_id + "_"
        ));
        host->register_kernel(make_kernel<VIUWSO, RightScalarMultiply, T>(
                "inplace_right_scalar_multiply_" + type_id + "_"
        ));

        host->register_kernel(
                make_kernel<VBO, Add, T>("addition_" + type_id + "_")
        );
        host->register_kernel(
                make_kernel<VIBO, Add, T>("inplace_addition_" + type_id + "_")
        );
        host->register_kernel(
                make_kernel<VBO, Sub, T>("subtraction_" + type_id + "_")
        );
        host->register_kernel(make_kernel<VIBO, Sub, T>(
                "inplace_subtraction_" + type_id + "_"
        ));

        host->register_kernel(
                make_kernel<VIBWSO, FusedLeftScalarMultiplyAdd, T>(
                        "fused_add_scalar_left_mul_" + type_id + "_"
                )
        );
        host->register_kernel(
                make_kernel<VIBWSO, FusedRightScalarMultiplyAdd, T>(
                        "fused_add_scalar_right_mul_" + type_id + "_"
                )
        );

        host->register_kernel(
                make_kernel<VIBWSO, FusedLeftScalarMultiplySub, T>(
                        "fused_sub_scalar_left_mul_" + type_id + "_"
                )
        );
        host->register_kernel(
                make_kernel<VIBWSO, FusedRightScalarMultiplySub, T>(
                        "fused_sub_scalar_right_mul_" + type_id + "_"
                )
        );
    }

public:
    RegisterHostKernels()
    {
        auto host = devices::get_host_device();

        register_all_kernels<float>(host);
        register_all_kernels<double>(host);
    }
};

}// namespace

optional<Kernel> algebra::dtl::get_kernel(
        string_view kernel_name,
        string_view type_id,
        string_view suffix,
        const Device& device
)
{
    static const RegisterHostKernels _registered_kernels;
    auto name = string(kernel_name) + '_' + string(type_id) + '_'
            + string(suffix);
    return device->get_kernel(name);
}

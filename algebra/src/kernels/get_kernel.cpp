//
// Created by sam on 4/17/24.
//

#include "kernels/kernel.h"
#include "roughpy/devices/host_device.h"

#include <roughpy/core/types.h>

// #include "host_kernels/vector_add.h"
// #include "host_kernels/vector_inplace_add.h"
// #include "host_kernels/vector_inplace_fused_left_scalar_multiply_add.h"
// #include "host_kernels/vector_inplace_fused_left_scalar_multiply_sub.h"
// #include "host_kernels/vector_inplace_fused_right_scalar_divide_add.h"
// #include "host_kernels/vector_inplace_fused_right_scalar_divide_sub.h"
// #include "host_kernels/vector_inplace_fused_right_scalar_multiply_add.h"
// #include "host_kernels/vector_inplace_fused_right_scalar_multiply_sub.h"
// #include "host_kernels/vector_inplace_left_scalar_multiply.h"
// #include "host_kernels/vector_inplace_right_scalar_multiply.h"
// #include "host_kernels/vector_inplace_sub.h"
// #include "host_kernels/vector_left_scalar_multiply.h"
// #include "host_kernels/vector_right_scalar_divide.h"
// #include "host_kernels/vector_right_scalar_multiply.h"
// #include "host_kernels/vector_sub.h"
// #include "host_kernels/vector_uminus.h"

#include "host_kernels/vector_binary_operator.h"
#include "host_kernels/vector_unary_operator.h"
#include <roughpy/device_support/operators.h>

using namespace rpy;
using namespace rpy::devices;

namespace rpy {
namespace algebra {
namespace dtl {}// namespace dtl
}// namespace algebra
}// namespace rpy

namespace {

class RegisterHostKernels
{
    template <typename T>
    using UminusOp = algebra::VectorUnaryOperator<operators::Uminus, T>;

    template <typename T>
    using LeftScalarMulOp = algebra::
            VectorUnaryWithScalarOperator<operators::LeftScalarMultiply, T>;

    template <typename T>
    using RightScalarMulOp = algebra::
            VectorUnaryWithScalarOperator<operators::RightScalarMultiply, T>;

    template <typename T>
    using AdditionOp = algebra::VectorBinaryOperator<operators::Add, T>;

    template <typename T>
    using SubtractionOp = algebra::VectorBinaryOperator<operators::Sub, T>;

    template <typename T>
    using LeftInplaceScalarMulOp
            = algebra::VectorInplaceUnaryWithScalarOperator<
                    operators::LeftScalarMultiply,
                    T>;

    template <typename T>
    using RightInplaceScalarMulOp
            = algebra::VectorInplaceUnaryWithScalarOperator<
                    operators::RightScalarMultiply,
                    T>;

    template <typename T>
    using InplaceAdditionOp
            = algebra::VectorInplaceBinaryOperator<operators::Add, T>;

    template <typename T>
    using InplaceSubtractionOp
            = algebra::VectorInplaceBinaryOperator<operators::Sub, T>;

    template <typename T>
    using FusedLeftScalarMultiplyAddOp
            = algebra::VectorInplaceBinaryWithScalarOperator<
                    operators::FusedLeftScalarMultiplyAdd,
                    T>;

    template <typename T>
    using FusedRightScalarMultiplyAddOp
            = algebra::VectorInplaceBinaryWithScalarOperator<
                    operators::FusedRightScalarMultiplyAdd,
                    T>;

    template <typename T>
    using FusedLeftScalarMultiplySubOp
            = algebra::VectorInplaceBinaryWithScalarOperator<
                    operators::FusedLeftScalarMultiplySub,
                    T>;

    template <typename T>
    using FusedRightScalarMultiplySubOp
            = algebra::VectorInplaceBinaryWithScalarOperator<
                    operators::FusedRightScalarMultiplySub,
                    T>;

    template <typename T>
    void register_all_kernels(const HostDevice& host)
    {
        UminusOp<T>::register_kernels(host);

        LeftScalarMulOp<T>::register_kernels(host);
        RightScalarMulOp<T>::register_kernels(host);

        AdditionOp<T>::register_kernels(host);
        SubtractionOp<T>::register_kernels(host);

        LeftInplaceScalarMulOp<T>::register_kernels(host);
        RightInplaceScalarMulOp<T>::register_kernels(host);

        InplaceAdditionOp<T>::register_kernels(host);
        InplaceSubtractionOp<T>::register_kernels(host);

        FusedLeftScalarMultiplyAddOp<T>::register_kernels(host);
        FusedRightScalarMultiplyAddOp<T>::register_kernels(host);
        FusedLeftScalarMultiplySubOp<T>::register_kernels(host);
        FusedRightScalarMultiplySubOp<T>::register_kernels(host);
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
    return device->get_kernel(string_cat(kernel_name, '_', type_id, '_', suffix)
    );
}

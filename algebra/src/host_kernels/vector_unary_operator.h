//
// Created by sam on 4/30/24.
//

#ifndef VECTOR_UNARY_OPERATOR_H
#define VECTOR_UNARY_OPERATOR_H

#include <roughpy/core/ranges.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/types.h>

#include <roughpy/devices/device_handle.h>

#include <roughpy/device_support/host_kernel.h>

namespace rpy {
namespace algebra {

namespace dtl {

template <typename T>
struct IdOfTypeImpl;

template <typename T>
constexpr string_view IdOfType = IdOfTypeImpl<T>::value;

template <typename Op>
constexpr string_view NameOfOperator = Op::name;

template <typename T, typename Op>
struct VectorUnaryOperator {
    struct Dd;
    struct Ds;

    static void register_kernels(const devices::Device& device);
};

template <typename T, typename Op>
struct VectorUnaryWithScalarOperator {
    struct Ddv;
    struct Dsv;

    static void register_kernels(const devices::Device& device);
};

template <typename T, typename Op>
struct VectorInplaceUnaryOperator {
    struct D;
    struct S;

    static void register_kernels(const devices::Device& device);
};

template <typename T, typename Op>
struct VectorInplaceUnaryWithScalarOperator {
    struct Dv;
    struct Sv;

    static void register_kernels(const devices::Device& device);
};

}// namespace dtl

/*
 * All that follows are the details of the implementation.
 */

namespace dtl {

template <typename Op, typename T>
void vector_kernel_eval_Dd(Op&& op, Slice<T> out, Slice<const T> in)
{
    for (auto&& [oscal, iscal] : views::zip(out, in)) { oscal = op(iscal); }
}

template <typename Op, typename T>
void vector_kernel_eval_Ds(
        Op&& op,
        Slice<T> out,
        Slice<const dimn_t> in_keys,
        Slice<const T> in
)
{
    for (auto&& [i, val] : views::zip(in_keys, in)) { out[i] = op(val); }
}

template <typename Op, typename T>
void vector_inplace_kernel_eval_D(Op&& op, Slice<T> arg)
{
    for (auto&& val : arg) { val = op(arg); }
}

template <typename T, typename Op>
struct VectorUnaryOperator<T, Op>::Dd {
    void operator()(Slice<T> out, Slice<const T> in) const
    {
        vector_kernel_eval_Dd(Op(), out, in);
    }
};

template <typename T, typename Op>
struct VectorUnaryOperator<T, Op>::Ds {
    void
    operator()(Slice<T> out, Slice<const dimn_t> in_keys, Slice<const T> in)
            const
    {
        vector_kernel_eval_Ds(Op(), out, in_keys, in);
    }
};

template <typename T, typename Op>
void VectorUnaryOperator<T, Op>::register_kernels(const devices::Device& device)
{
    using devices::HostKernel;
    const string name = NameOfOperator<Op>;

    device->register_kernel(
            Kernel(new HostKernel<Dd>(name + '_' + IdOfType<T> + '_' + "Dd"))
    );
    device->register_kernel(
            Kernel(new HostKernel<Ds>(name + '_' + IdOfType<T> + '_' + "Ds"))
    );
}

template <typename T, typename Op>
struct VectorUnaryWithScalarOperator<T, Op>::Ddv {
    void operator()(Slice<T> out, Slice<const T> in, const T& scal) const
    {
        vector_kernel_eval_Dd(Op(scal), out, in);
    }
};

template <typename T, typename Op>
struct VectorUnaryWithScalarOperator<T, Op>::Dsv {
    void operator()(
            Slice<T> out,
            Slice<const dimn_t> in_keys,
            Slice<const T> in,
            const T& scal
    ) const
    {
        vector_kernel_eval_Dd(Op(scal), out, in_keys, in);
    }
};

template <typename T, typename Op>
void VectorUnaryWithScalarOperator<T, Op>::register_kernels(
        const devices::Device& device
)
{
    using devices::HostKernel;
    const string name = NameOfOperator<Op>;

    device->register_kernel(
            Kernel(new HostKernel<Ddv>(name + '_' + IdOfType<T> + '_' + "Ddv"))
    );
    device->register_kernel(
            Kernel(new HostKernel<Dsv>(name + '_' + IdOfType<T> + '_' + "Dsv"))
    );
}

template <typename T, typename Op>
struct VectorInplaceUnaryOperator<T, Op>::D {
    void operator()(Slice<T> arg) { vector_inplace_kernel_eval_D(Op(), arg); }
};

template <typename T, typename Op>
struct VectorInplaceUnaryOperator<T, Op>::S {
    void operator()(Slice<dimn_t> RPY_UNUSED_VAR keys, Slice<T> arg)
    {
        vector_inplace_kernel_eval_D(Op(), arg);
    }
};

template <typename T, typename Op>
void VectorInplaceUnaryOperator<T, Op>::register_kernels(
        const devices::Device& device
)
{
    using devices::HostKernel;
    const string name = NameOfOperator<Op>;

    device->register_kernel(
            Kernel(new HostKernel<D>(name + '_' + IdOfType<T> + '_' + "D"))
    );
    device->register_kernel(
            Kernel(new HostKernel<S>(name + '_' + IdOfType<T> + '_' + "S"))
    );
}

template <typename T, typename Op>
struct VectorInplaceUnaryWithScalarOperator<T, Op>::Dv {
    void operator()(Slice<T> arg, const T& scalar)
    {
        vector_inplace_kernel_eval_D(Op(scalar), arg);
    }
};

template <typename T, typename Op>
struct VectorInplaceUnaryWithScalarOperator<T, Op>::Sv {
    void operator()(
            Slice<dimn_t> RPY_UNUSED_VAR keys,
            Slice<T> values,
            const T& scal
    )
    {
        vector_inplace_kernel_eval_D(Op(scal), values);
    }
};

template <typename T, typename Op>
void VectorInplaceUnaryWithScalarOperator<T, Op>::register_kernels(
        const devices::Device& device
)
{

    using devices::HostKernel;
    const string name = NameOfOperator<Op>;

    device->register_kernel(
            Kernel(new HostKernel<Dv>(name + '_' + IdOfType<T> + '_' + "Dv"))
    );
    device->register_kernel(
            Kernel(new HostKernel<Sv>(name + '_' + IdOfType<T> + '_' + "Sv"))
    );
}

}// namespace dtl

}// namespace algebra
}// namespace rpy

#endif// VECTOR_UNARY_OPERATOR_H

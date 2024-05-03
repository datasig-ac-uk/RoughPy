//
// Created by sam on 4/30/24.
//

#ifndef VECTOR_BINARY_OPERATOR_H
#define VECTOR_BINARY_OPERATOR_H

#include "vector_binary_operator.h"

#include <roughpy/core/ranges.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/traits.h>
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

template <typename Op, typename T>
void vector_kernel_eval_Ddd(
        Op&& op,
        Slice<T> out,
        Slice<const T> left,
        Slice<const T> right
);

template <typename T, typename Op>
struct VectorBinaryOperator {
    struct Ddd;
    struct Dds;
    struct Dsd;
    struct Dss;

    static void register_kernels(const devices::Device& device);
};

template <typename T, typename Op>
struct VectorBinaryWithScalarOperator {
    struct Dddv;
    struct Ddsv;
    struct Dsdv;
    struct Dssv;

    static void register_kernels(const devices::Device& device);
};

template <typename T, typename Op>
struct VectorInplaceBinaryOperator {
    struct Dd;
    struct Ds;

    static void register_kernels(const devices::Device& device);
};

template <typename T, typename Op>
struct VectorInplaceBinaryWithScalarOperator {
    struct Ddv;
    struct Dsv;

    static void register_kernels(const devices::Device& device);
};

}// namespace dtl

template <template <typename...> class Operator, typename T>
using VectorBinaryOperator = dtl::VectorBinaryOperator<T, Operator<T>>;

template <template <typename...> class Operator, typename T>
using VectorBinaryWithScalarOperator
        = dtl::VectorBinaryWithScalarOperator<T, Operator<T>>;

template <template <typename...> class Operator, typename T>
using VectorInplaceBinaryOperator
        = dtl::VectorInplaceBinaryOperator<T, Operator<T>>;

template <template <typename...> class Operator, typename T>
using VectorInplaceBinaryWithScalarOperator
        = dtl::VectorInplaceBinaryWithScalarOperator<T, Operator<T>>;

/*
 * Now for all the details.
 */

namespace dtl {

template <typename Op, typename T>
void vector_kernel_eval_Ddd(
        Op&& op,
        Slice<T> out,
        Slice<const T> left,
        Slice<const T> right
)
{
    for (auto&& [oscal, lscal, rscal] : views::zip(out, left, right)) {
        oscal = op(lscal, rscal);
    }
}

template <typename Op, typename T>
void vector_kernel_eval_Dds(
        Op&& op,
        Slice<T> out,
        Slice<const T> left,
        Slice<const dimn_t> right_keys,
        Slice<const T> right
)
{
    constexpr T zero;
    for (auto&& [oscal, lscal] : views::zip(out, left)) {
        oscal = op(lscal, zero);
    }
    for (auto&& [i, rscal] : views::zip(right_keys, right)) {
        out[i] += op(zero, rscal);
    }
}

template <typename Op, typename T>
void vector_kernel_eval_Dsd(
        Op&& op,
        Slice<T> out,
        Slice<const dimn_t> left_keys,
        Slice<const T> left,
        Slice<const T> right
)
{
    constexpr T zero;
    for (auto&& [oscal, rscal] : views::zip(out, right)) {
        oscal = op(zero, rscal);
    }
    for (auto&& [i, lscal] : views::zip(left_keys, left)) {
        out[i] += op(lscal, zero);
    }
}

template <typename Op, typename T>
void vector_inplace_kernel_eval_Dd(Op&& op, Slice<T> out, Slice<const T> right)
{
    for (auto&& [oscal, rscal] : views::zip(out, right)) {
        oscal = op(oscal, rscal);
    }
}
template <typename Op, typename T>
void vector_inplace_kernel_eval_Ds(
        Op&& op,
        Slice<T> out,
        Slice<const dimn_t> right_keys,
        Slice<const T> right
)
{
    constexpr T zero;
    for (auto&& [i, rscal] : views::zip(right_keys, right)) {
        out[i] += op(zero, rscal);
    }
}

template <typename Op, typename T>
void vector_kernel_eval_Dss(
        Op&& op,
        Slice<T> out,
        Slice<const dimn_t> left_keys,
        Slice<const T> left,
        Slice<const dimn_t> right_keys,
        Slice<const T> right
)
{
    constexpr T zero;
    for (auto&& [i, lscal] : views::zip(left_keys, left)) {
        out[i] += op(lscal, zero);
    }
    for (auto&& [i, rscal] : views::zip(right_keys, right)) {
        out[i] += op(zero, rscal);
    }
}

template <typename Op, typename T>
struct VectorBinaryOperator<Op, T>::Ddd {
    void
    operator()(Slice<T> out, Slice<const T> left, Slice<const T> right) const
    {
        Op op;
        for (auto&& [oscal, lscal, rscal] : views::zip(out, left, right)) {
            oscal = op(lscal, rscal);
        }
    }
};

template <typename Op, typename T>
struct VectorBinaryOperator<Op, T>::Dds {
    void operator()(
            Slice<T> out,
            Slice<const T> left,
            Slice<const dimn_t> right_keys,
            Slice<const T> right
    ) const
    {
        Op op;
        T zero;
        for (auto&& [oscal, lscal] : views::zip(out, left)) {
            oscal = op(lscal, zero);
        }
        for (auto&& [i, rscal] : views::zip(right_keys, right)) {
            out[i] += op(zero, rscal);
        }
    }
};

template <typename Op, typename T>
struct VectorBinaryOperator<Op, T>::Dsd {
    void operator()(
            Slice<T> out,
            Slice<const dimn_t> left_keys,
            Slice<const T> left,
            Slice<const T> right
    ) const
    {
        Op op;
        T zero;
        for (auto&& [oscal, rscal] : views::zip(out, right)) {
            oscal = op(zero, rscal);
        }
        for (auto&& [i, lscal] : views::zip(left_keys, left)) {
            out[i] += op(lscal, zero);
        }
    }
};

template <typename Op, typename T>
struct VectorBinaryOperator<Op, T>::Dss {
    void operator()(
            Slice<T> out,
            Slice<const dimn_t> left_keys,
            Slice<const T> left,
            Slice<const dimn_t> right_keys,
            Slice<const T> right
    ) const
    {
        Op op;
        T zero;
        for (auto&& [i, lscal] : views::zip(left_keys, left)) {
            out[i] += op(lscal, zero);
        }
        for (auto&& [i, rscal] : views::zip(right_keys, right)) {
            out[i] += op(zero, rscal);
        }
    }
};

template <typename T, typename Op>
void VectorBinaryOperator<T, Op>::register_kernels(const devices::Device& device
)
{
    using devices::HostKernel;
    const string name = NameOfOperator<Op>;

    device->register_kernel(
            Kernel(new HostKernel<Ddd>(name + '_' + IdOfType<T> + '_' + "Ddd"))
    );
    device->register_kernel(
            Kernel(new HostKernel<Dds>(name + '_' + IdOfType<T> + '_' + "Dds"))
    );
    device->register_kernel(
            Kernel(new HostKernel<Dsd>(name + '_' + IdOfType<T> + '_' + "Dsd"))
    );
    device->register_kernel(
            Kernel(new HostKernel<Dss>(name + '_' + IdOfType<T> + '_' + "Dss"))
    );
}

template <typename Op, typename T>
struct VectorBinaryWithScalarOperator<Op, T>::Dddv {
    void operator()(
            Slice<T> out,
            Slice<const T> left,
            Slice<const T> right,
            const T& scal
    ) const
    {
        vector_kernel_eval_Ddd(Op(scal), out, left, right);
    }
};

template <typename Op, typename T>
struct VectorBinaryWithScalarOperator<Op, T>::Ddsv {
    void operator()(
            Slice<T> out,
            Slice<const T> left,
            Slice<const dimn_t> right_keys,
            Slice<const T> right,
            const T& scal
    ) const
    {
        vector_kernel_eval_Dds(Op(scal), out, left, right_keys, right);
    }
};

template <typename Op, typename T>
struct VectorBinaryWithScalarOperator<Op, T>::Dsdv {
    void operator()(
            Slice<T> out,
            Slice<const dimn_t> left_keys,
            Slice<const T> left,
            Slice<const T> right,
            const T& scal
    ) const
    {
        vector_kernel_eval_Dsd(Op(scal), out, left_keys, left, right);
    }
};

template <typename Op, typename T>
struct VectorBinaryWithScalarOperator<Op, T>::Dssv {
    void operator()(
            Slice<T> out,
            Slice<const dimn_t> left_keys,
            Slice<const T> left,
            Slice<const dimn_t> right_keys,
            Slice<const T> right,
            const T& scal
    ) const
    {
        vector_kernel_eval_Dss(
                Op(scal),
                out,
                left_keys,
                left,
                right_keys,
                right
        );
    }
};

template <typename T, typename Op>
void VectorBinaryWithScalarOperator<T, Op>::register_kernels(
        const devices::Device& device
)
{
    using devices::HostKernel;
    const string name = NameOfOperator<Op>;

    device->register_kernel(Kernel(
            new HostKernel<Dddv>(name + '_' + IdOfType<T> + '_' + "Dddv")
    ));
    device->register_kernel(Kernel(
            new HostKernel<Ddsv>(name + '_' + IdOfType<T> + '_' + "Ddsv")
    ));
    device->register_kernel(Kernel(
            new HostKernel<Dsdv>(name + '_' + IdOfType<T> + '_' + "Dsdv")
    ));
    device->register_kernel(Kernel(
            new HostKernel<Dssv>(name + '_' + IdOfType<T> + '_' + "Dssv")
    ));
}

template <typename Op, typename T>
struct VectorInplaceBinaryOperator<Op, T>::Dd {
    void operator()(Slice<T> out, Slice<const T> right) const
    {
        vector_inplace_kernel_eval_Dd(Op(), out, right);
    }
};

template <typename Op, typename T>
struct VectorInplaceBinaryOperator<Op, T>::Ds {
    void operator()(
            Slice<T> out,
            Slice<const dimn_t> right_keys,
            Slice<const T> right
    ) const
    {
        vector_inplace_kernel_eval_Ds(Op(), out, right_keys, right);
    }
};

template <typename T, typename Op>
void VectorInplaceBinaryOperator<T, Op>::register_kernels(
        const devices::Device& device
)
{
    using devices::HostKernel;
    const string name = NameOfOperator<Op>;

    device->register_kernel(
            Kernel(new HostKernel<Dd>(name + '_' + IdOfType<T> + '_' + "Dddv"))
    );
    device->register_kernel(
            Kernel(new HostKernel<Ds>(name + '_' + IdOfType<T> + '_' + "Ddsv"))
    );
}

template <typename T, typename Op>
struct VectorInplaceBinaryWithScalarOperator<T, Op>::Ddv {
    void operator()(Slice<T> out, Slice<const T> right, const T& scal) const
    {
        vector_inplace_kernel_eval_Dd(Op(scal), out, right);
    }
};

template <typename T, typename Op>
struct VectorInplaceBinaryWithScalarOperator<T, Op>::Dsv {
    void operator()(
            Slice<T> out,
            Slice<const dimn_t> right_keys,
            Slice<const T> right,
            const T& scal
    ) const
    {
        vector_inplace_kernel_eval_Ds(Op(scal), out, right_keys, right);
    }
};

template <typename T, typename Op>
void VectorInplaceBinaryWithScalarOperator<T, Op>::register_kernels(
        const devices::Device& device
)
{
    using devices::HostKernel;
    const string name = NameOfOperator<Op>;

    device->register_kernel(
            Kernel(new HostKernel<Ddv>(name + '_' + IdOfType<T> + '_' + "Dddv"))
    );
    device->register_kernel(
            Kernel(new HostKernel<Dsv>(name + '_' + IdOfType<T> + '_' + "Ddsv"))
    );
}

}// namespace dtl

}// namespace algebra
}// namespace rpy

#endif// VECTOR_BINARY_OPERATOR_H

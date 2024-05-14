//
// Created by sam on 4/30/24.
//

#ifndef VECTOR_BINARY_OPERATOR_H
#define VECTOR_BINARY_OPERATOR_H

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

template <typename T, typename Op>
struct VectorBinaryOperator {
    struct Ddd;
    struct Dds;
    struct Dsd;
    struct Dss;
    struct Sdd;
    struct Sds;
    struct Ssd;
    struct Sss;

    static void register_kernels(const devices::Device& device);
};

template <typename T, typename Op>
struct VectorBinaryWithScalarOperator {
    struct Dddv;
    struct Ddsv;
    struct Dsdv;
    struct Dssv;
    struct Sddv;
    struct Sdsv;
    struct Ssdv;
    struct Sssv;

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

/*
 * This is the easy case, we only need to make sure we loop
 * over all of the the common elements and then over any terms
 * that are only present in one of the terms.
 *
 * The only tricky bit is to only add the terms that are noo-zero.
 */
template <typename T>
void vector_kernel_sparse_write(
        Slice<dimn_t>& out_keys,
        Slice<T>& out,
        dimn_t& out_i,
        const dimn_t& result_i,
        T&& result,
        const T& zero
)
{
    if (result != zero) {
        RPY_DBG_ASSERT(out_i < out_keys.size());
        out_keys[out_i] = result_i;
        out[out_i] = std::move(result);
        ++out_i;
    }
}

template <typename Op, typename T>
void vector_kernel_eval_Sdd(
        Op&& op,
        Slice<dimn_t> out_keys,
        Slice<T> out,
        Slice<const T> left,
        Slice<const T> right
)
{
    constexpr T zero{};

    dimn_t out_i = 0;
    ;
    dimn_t i = 0;
    for (; i < std::min(left.size(), right.size()); ++i) {
        vector_kernel_sparse_write(
                out_keys,
                out,
                out_i,
                i,
                op(left[i], right[i]),
                zero
        );
    }

    for (; i < left.size(); ++i) {
        vector_kernel_sparse_write(
                out_keys,
                out,
                out_i,
                i,
                op(left[i], zero),
                zero
        );
    }

    for (; i < right.size(); ++i) {
        vector_kernel_sparse_write(
                out_keys,
                out,
                out_i,
                i,
                op(zero, right[i]),
                zero
        );
    }
}

/*
 * The next two cases are similar. The tricky part is to looping over everything
 * because th sparse terms might out outstretch the dense terms.
 */
template <typename Op, typename T>
void vector_kernel_eval_Sds(
        Op&& op,
        Slice<dimn_t> out_keys,
        Slice<T> out,
        Slice<const T> left,
        Slice<const dimn_t> right_keys,
        Slice<const T> right
)
{
    constexpr T zero{};

    dimn_t i = 0;
    dimn_t out_i = 0;
    auto rit = ranges::begin(right);
    auto rkit = ranges::begin(right_keys);
    const auto rkend = ranges::end(right_keys);

    // Let's just check if the range is sorted
    if (ranges::is_sorted(right_keys)) {
        for (; i < left.size(); ++i) {
            if (rkit != rkend) {
                vector_kernel_sparse_write(
                        out_keys,
                        out,
                        out_i,
                        i,
                        op(left[i], (*rkit == i) ? *rit : zero),
                        zero
                );
                ++rkit;
                ++rit;
            } else {
                vector_kernel_sparse_write(
                        out_keys,
                        out,
                        out_i,
                        i,
                        op(left[i], zero),
                        zero
                );
            }
        }
    } else {
        const auto rkbegin = ranges::begin(right_keys);

        for (; i < left.size(); ++i) {
            if ((rit = ranges::find(rkbegin, rkend, i)) != rkend) {
                vector_kernel_sparse_write(
                        out_keys,
                        out,
                        out_i,
                        i,
                        op(left[i], right[ranges::distance(rkbegin, rit)]),
                        zero
                );
            } else {
                vector_kernel_sparse_write(
                        out_keys,
                        out,
                        out_i,
                        i,
                        op(left[i], zero),
                        zero
                );
            }
        }

        auto filter = [max = left.size()](const auto& val) {
            return std::get<0>(val) >= max;
        };
        for (const auto& [rkey, rval] :
             views::zip(right_keys, right) | views::filter(filter)) {
            vector_kernel_sparse_write(
                    out_keys,
                    out,
                    out_i,
                    i,
                    op(zero, rval),
                    zero
            );
        }
    }
}

template <typename Op, typename T>
void vector_kernel_eval_Ssd(
        Op&& op,
        Slice<dimn_t> out_keys,
        Slice<T> out,
        Slice<const dimn_t> left_keys,
        Slice<const T> left,
        Slice<const T> right
)
{
    constexpr T zero{};

    dimn_t i = 0;
    dimn_t out_i = 0;
    auto lit = ranges::begin(left);
    auto lkit = ranges::begin(left_keys);
    const auto lkend = ranges::end(left_keys);

    // Let's just check if the range is sorted
    if (ranges::is_sorted(left_keys)) {
        for (; i < right.size(); ++i) {
            if (lkit != lkend) {
                vector_kernel_sparse_write(
                        out_keys,
                        out,
                        out_i,
                        i,
                        op((*lkit == i) ? *lit : zero, right[i]),
                        zero
                );
                ++lkit;
                ++lit;
            } else {
                vector_kernel_sparse_write(
                        out_keys,
                        out,
                        out_i,
                        i,
                        op(zero, right[i]),
                        zero
                );
            }
        }

        for (; lkit != lkend; ++lkit, ++lit) {
            vector_kernel_sparse_write(
                    out_keys,
                    out,
                    out_i,
                    i,
                    op(*lit, zero),
                    zero
            );
        }
    } else {
        const auto lkbegin = ranges::begin(left_keys);

        for (; i < right.size(); ++i) {
            if ((lit = ranges::find(lkbegin, lkend, i)) != lkend) {
                vector_kernel_sparse_write(
                        out_keys,
                        out,
                        out_i,
                        i,
                        op(left[ranges::distance(lkbegin, lit)], right[i]),
                        zero
                );
            } else {
                vector_kernel_sparse_write(
                        out_keys,
                        out,
                        out_i,
                        i,
                        op(zero, right[i]),
                        zero
                );
            }
        }

        auto filter = [max = right.size()](const auto& val) {
            return std::get<0>(val) >= max;
        };
        for (const auto& [lkey, lval] :
             views::zip(left_keys, left) | views::filter(filter)) {
            vector_kernel_sparse_write(
                    out_keys,
                    out,
                    out_i,
                    i,
                    op(lval, zero),
                    zero
            );
        }
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

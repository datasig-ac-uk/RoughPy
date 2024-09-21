//
// Created by sam on 19/09/24.
//

#ifndef FREETENSORMULTIPLICATION_H
#define FREETENSORMULTIPLICATION_H

#include "TensorMultiplication.h"
#include "scalar_vector.h"

#include <roughpy/device_support/host_kernel.h>

namespace params = rpy::devices::params;

namespace rpy {
namespace algebra {

template <template <typename> class Operation, typename T>
class StandardFreeTensorMultiplication
    : public devices::dtl::HostKernelBase<
              StandardFreeTensorMultiplication<Operation, T>,
              params::ResultBuffer<T>,
              params::Buffer<T>,
              params::Buffer<T>,
              params::Operator<T>,
              params::Buffer<dimn_t>,
              params::Buffer<dimn_t>,
              params::Buffer<dimn_t>>
{

    using base_t = devices::dtl::HostKernelBase<
            StandardFreeTensorMultiplication,
            params::ResultBuffer<T>,
            params::Buffer<T>,
            params::Buffer<T>,
            params::Operator<T>,
            params::Buffer<dimn_t>,
            params::Buffer<dimn_t>,
            params::Buffer<dimn_t>>;

public:
    using base_t::base_t;

    RPY_NO_DISCARD static string_view get_base_name() noexcept
    {
        static string name = string_cat(
                "free_tensor_multiplication_",
                Operation<devices::Value>::name
        );
        return name;
    }

    RPY_NO_DISCARD static string_view get_name() noexcept
    {
        static string name
                = string_cat(get_base_name(), "_", devices::type_id_of<T>);
        return name;
    }

    struct Impl {
        devices::Size3 localDim;
        devices::Size3 globalDim;

        void operator()(
                Slice<T> result,
                Slice<const T> lhs,
                Slice<const T> rhs,
                Operation<T>&& op,
                Slice<const dimn_t> powers,
                deg_t degree
        ) const
        {}
    };

    static void
    run(const devices::KernelLaunchParams& params,
        Slice<T> result,
        Slice<const T> lhs,
        Slice<const T> rhs,
        Operation<T>&& op,
        Slice<const dimn_t> powers,
        Slice<const dimn_t> offsets,
        deg_t max_degree,
        deg_t min_lhs_deg,
        deg_t min_rhs_deg)
    {
        const auto& work_dims = params.work_dims();
        const auto& group_size = params.work_groups();

        for (deg_t degree = max_degree; degree > 0; --degree) {

            for (dimn_t iz = 0; iz < group_size.x; ++iz) {
                for (dimn_t iy = 0; iy < group_size.y; ++iy) {
                    for (dimn_t ix = 0; ix < group_size.z; ++ix) {
                        Impl func{
                                devices::Size3(work_dims),
                                {ix, iy, iz}
                        };

                        func(result, lhs, rhs, op, powers, degree);
                    }
                }
            }
        }
    }
};

class FreeTensorMultiplication : TensorMultiplication
{

public:
    using TensorMultiplication::TensorMultiplication;

protected:
    RPY_NO_DISCARD devices::KernelLaunchParams
    get_launch_params(const devices::KernelArguments& args) const noexcept;

public:
    using TensorMultiplication::antipode;

    void
    eval(scalars::ScalarVector& destination,
         const scalars::ScalarVector& left,
         const scalars::ScalarVector& right,
         const devices::operators::Operator& op,
         deg_t max_degree,
         deg_t min_left_degree,
         deg_t min_right_degree) const;

    void eval_inplace(
            scalars::ScalarVector& left,
            const scalars::ScalarVector& right,
            const devices::operators::Operator& op,
            deg_t max_degree,
            deg_t min_left_degree,
            deg_t min_right_degree
    ) const;
};

}// namespace algebra
}// namespace rpy

#endif// FREETENSORMULTIPLICATION_H

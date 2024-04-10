//
// Created by sam on 10/04/24.
//

#ifndef KERNEL_H
#define KERNEL_H

#include <roughpy/core/macros.h>
#include <roughpy/core/ranges.h>
#include <roughpy/core/types.h>

#include <roughpy/scalars/devices/core.h>
#include <roughpy/scalars/devices/device_handle.h>
#include <roughpy/scalars/devices/kernel.h>

#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_array.h>
#include <roughpy/scalars/scalar_type.h>

#include "vector.h"

namespace rpy {
namespace algebra {

namespace dtl {

struct ArgumentCollector
{

};

}// namespace dtl

class VectorKernelBase
{

protected:
    using result_type = std::unique_ptr<VectorData>;

    template <typename... Args>
    RPY_NO_DISCARD static result_type
    make_result(const VectorData& arg, const Args&... args)
    {
        const auto arg_size = arg.size();
        auto result
                = std::make_unique<VectorData>(arg.scalars().type(), arg_size);

        if (!arg.keys().empty()) { result->mut_keys() = KeyArray(arg_size); }

        return result;
    }

    optional<devices::Kernel> get_kernel(
            string_view name,
            devices::Device device,
            const devices::Type* type
    ) const;
};

template <typename Derived>
class VectorKernelCommonBase : public VectorKernelBase
{
protected:
    template <typename... Args>
    static const scalars::ScalarType*
    get_type(const VectorData& data, const Args&... args) noexcept
    {
        return data.scalars().type();
    }
    template <typename... Args>
    static devices::Device
    get_device(const VectorData& data, const Args&... args) noexcept
    {
        return data.scalars().device();
    }

    RPY_NO_DISCARD const Derived& instance() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }
    RPY_NO_DISCARD Derived& instance() noexcept
    {
        return static_cast<Derived&>(*this);
    }
    template <typename... Args>
    bool eval_on_device(
            const devices::KernelLaunchParams& params,
            devices::Device device,
            const devices::Type* type,
            Args&&... args
    ) const;

    template <typename... Args>
    bool eval_on_host(
            const devices::KernelLaunchParams& params,
            const devices::Type* type,
            Args&&... args
    ) const
    {
    }
};

template <typename Derived>
class OutOfPlaceVectorKernelCommonBase : public VectorKernelCommonBase<Derived>
{
    using common_base_t = VectorKernelCommonBase<Derived>;

public:
    using typename common_base_t::result_type;

    template <typename... Args>
    RPY_NO_DISCARD result_type operator()(Args&&... args) const
    {
        auto result = common_base_t::make_result(args...);
        const auto& inst = this->instance();

        if (!inst.eval_on_device(*result, args...)) {
            if (!inst.eval_on_device(*result, args...)) {
                inst.eval_generic(*result, args...);
            }
        }

        return result;
    }
};

template <typename Derived>
class InplaceVectorKernelCommonBase : public VectorKernelCommonBase<Derived>
{
    using common_base_t = VectorKernelCommonBase<Derived>;

public:
    using result_type = void;

    template <typename... Args>
    bool eval_on_device(VectorData& out, const Args&... args) const
    {
        return false;
    }
    template <typename... Args>
    bool eval_on_host(VectorData& out, const Args&... args) const
    {
        return false;
    }
    template <typename... Args>
    void eval_generic(VectorData& out, const Args&... args) const
    {}

    template <typename... Args>
    void operator()(VectorData& result, Args&&... args) const
    {
        const auto& inst = this->instance();
        if (!inst.eval_on_device(result, args...)) {
            if (!inst.eval_on_device(result, args...)) {
                inst.eval_generic(result, args...);
            }
        }
    }
};

template <typename Derived>
class UnaryVectorKernel : public OutOfPlaceVectorKernelCommonBase<Derived>
{
    using common_base_t = OutOfPlaceVectorKernelCommonBase<Derived>;

    static void eval_kernel(
            const devices::Kernel& kernel,
            const devices::KernelLaunchParams& params,
            scalars::ScalarArray& out,
            const scalars::ScalarArray& in
    )
    {
        kernel(params, out.mut_buffer(), in.buffer());
    }

public:
    bool eval_on_device(VectorData& out, const VectorData& arg) const;
    bool eval_on_host(VectorData& out, const VectorData& arg) const;
    void eval_generic(VectorData& out, const VectorData& arg) const;
};

template <typename Derived>
class UnaryInplaceVectorKernel : public VectorKernelCommonBase<Derived>
{
    static void eval_kernel(
            const devices::Kernel& kernel,
            const devices::KernelLaunchParams& params,
            scalars::ScalarArray& arg
    )
    {
        kernel(params, arg.mut_buffer());
    }

public:
    bool eval_on_device(VectorData& arg) const;
    bool eval_on_host(VectorData& arg) const;
    void eval_generic(VectorData& arg) const;
};

template <typename Derived>
class BinaryVectorKernel : public OutOfPlaceVectorKernelCommonBase<Derived>
{
    using common_base_t = OutOfPlaceVectorKernelCommonBase<Derived>;

public:
    bool eval_on_device(
            VectorData& out,
            const VectorData& left,
            const VectorData& right
    ) const;
    bool eval_on_host(
            VectorData& out,
            const VectorData& left,
            const VectorData& right
    ) const;
    bool eval_generic(
            VectorData& out,
            const VectorData& left,
            const VectorData& right
    ) const;
};

template <typename Derived>
template <typename... Args>
bool VectorKernelCommonBase<Derived>::eval_on_device(
        const devices::KernelLaunchParams& params,
        Args&&... args
) const
{}

template <typename Derived>
template <typename... Args>
bool VectorKernelCommonBase<Derived>::eval_on_host(
        const devices::KernelLaunchParams& params,
        Args&&... args
) const
{
}

template <typename Derived>
bool UnaryVectorKernel<Derived>::eval_on_device(
        VectorData& out,
        const VectorData& arg
) const
{
    const auto kernel_name = this->instance()->kernel_name();
    const auto device = arg.scalars().device();
    const auto scalar_type = arg.scalars().type();
    if (auto kernel
        = this->get_kernel(kernel_name, device, scalar_type->as_type())) {
        const devices::KernelLaunchParams params{
                devices::Size3(arg.size()),
                devices::Dim3(1)
        };
        eval_kernel(*kernel, params, out.mut_scalars(), arg.scalars());
        return true;
    }
    return false;
}
template <typename Derived>
bool UnaryVectorKernel<Derived>::eval_on_host(
        VectorData& out,
        const VectorData& arg
) const
{
    const auto kernel_name = this->instance()->kernel_name();
    const auto device = devices::get_host_device();
    const auto scalar_type = arg.scalars().type();
    if (auto kernel
        = this->get_kernel(kernel_name, device, scalar_type->as_type())) {
        const devices::KernelLaunchParams params{
                devices::Size3(arg.size()),
                devices::Dim3(1)
        };
        eval_kernel(
                *kernel,
                params,
                out.mut_scalars().mut_view(),
                arg.scalars().view()
        );
        return true;
    }
    return false;
}

template <typename Derived>
void UnaryVectorKernel<Derived>::eval_generic(
        VectorData& out,
        const VectorData& arg
) const
{
    const auto arg_size = arg.size();

    auto host_out = out.mut_scalars().mut_view();
    const auto host_in = arg.scalars().view();

    const auto& function = this->function().get_generic_function();

    for (dimn_t i = 0; i < arg_size; ++i) { function(host_out[i], host_in[i]); }
}

template <typename Derived>
bool UnaryInplaceVectorKernel<Derived>::eval_on_device(VectorData& arg) const
{

    return false;
}
template <typename Derived>
bool UnaryInplaceVectorKernel<Derived>::eval_on_host(VectorData& arg) const
{

    return false;
}
template <typename Derived>
void UnaryInplaceVectorKernel<Derived>::eval_generic(VectorData& arg) const
{
    const auto arg_size = arg.size();
    auto host_arg = arg.mut_scalars().mut_view();

    const auto& function = this->instance().get_generic_function();

    for (dimn_t i = 0; i < arg_size; ++i) { function(host_arg[i]); }
}

template <typename Derived>
bool BinaryVectorKernel<Derived>::eval_on_device(
        VectorData& out,
        const VectorData& left,
        const VectorData& right
) const
{
    return false;
}
template <typename Derived>
bool BinaryVectorKernel<Derived>::eval_on_host(
        VectorData& out,
        const VectorData& left,
        const VectorData& right
) const
{
    return false;
}
template <typename Derived>
bool BinaryVectorKernel<Derived>::eval_generic(
        VectorData& out,
        const VectorData& left,
        const VectorData& right
) const
{}

}// namespace algebra
}// namespace rpy

#endif// KERNEL_H

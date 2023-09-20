//
// Created by sam on 20/09/23.
//

#ifndef ROUGHPY_CPU_KERNEL_H
#define ROUGHPY_CPU_KERNEL_H

#include <roughpy/device/core.h>
#include <roughpy/device/kernel.h>

namespace rpy {
namespace device {

class CPUKernelInterface : public KernelInterface
{
    struct KernelInformation;

    struct Data {
        const void* fn_ptr;
        const KernelInformation* information;
    };

public:

    static inline void* create_data(const void* fn_ptr)
    {
        return new Data { fn_ptr,  };
    }

    static inline void* create_cl_data(void* cl_data)
    {
        return new Data { cl_data, nullptr };
    }

private:

    static bool is_cl_kernel(void* content) noexcept {
        return static_cast<Data*>(content)->information == nullptr;
    }

    static const void* fn_ptr(void* content) noexcept {
        RPY_DBG_ASSERT(!is_cl_kernel(content));
        return static_cast<Data*>(content)->fn_ptr;
    }

    static inline void* cl_kernel(void* content) noexcept {
        RPY_DBG_ASSERT(is_cl_kernel(is_cl_kernel(content)));
        return const_cast<void*>(static_cast<Data*>(content)->fn_ptr);
    }

    static inline const KernelInformation* information(void* content) noexcept
    {
        RPY_DBG_ASSERT(!is_cl_kernel(content));
        return static_cast<Data*>(content)->information;
    }

public:
    void* clone(void* content) const override;
    void clear(void* content) const override;
    string_view name(void* content) const override;
    dimn_t num_args(void* content) const override;
    Event launch_kernel_async(
            void* content, Queue queue, Slice<void*> args,
            Slice<dimn_t> arg_sizes, const KernelLaunchParams& params
    ) const override;

};

}// namespace device
}// namespace rpy

#endif// ROUGHPY_CPU_KERNEL_H

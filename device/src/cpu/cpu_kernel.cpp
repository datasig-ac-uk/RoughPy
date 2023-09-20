//
// Created by sam on 20/09/23.
//

#include "cpu_kernel.h"

#include "opencl/ocl_kernel.h"

using namespace rpy;
using namespace rpy::device;

struct CPUKernelInterface::KernelInformation {
    string name;
};

void* rpy::device::CPUKernelInterface::clone(void* content) const
{
    if (is_cl_kernel(content)) {
        return cl::kernel_interface()->clone(cl_kernel(content));
    } else {
        const auto* info = information(content);
        return new Data{fn_ptr(content), new KernelInformation{info->name}};
    }
}
void rpy::device::CPUKernelInterface::clear(void* content) const
{
    if (is_cl_kernel(content)) {
        cl::kernel_interface()->clear(cl_kernel(content));
    } else {
        delete const_cast<KernelInformation*>(information(content));
    }
    delete static_cast<Data*>(content);
}
std::string_view rpy::device::CPUKernelInterface::name(void* content) const
{
    if (is_cl_kernel(content)) {
        return cl::kernel_interface()->name(cl_kernel(content));
    } else {
        return information(content)->name;
    }
}
rpy::dimn_t rpy::device::CPUKernelInterface::num_args(void* content) const
{
    if (is_cl_kernel(content)) {
        return cl::kernel_interface()->num_args(cl_kernel(content));
    } else {
        return 0;
    }
}
rpy::device::Event rpy::device::CPUKernelInterface::launch_kernel_async(
        void* content, rpy::device::Queue queue, rpy::Slice<void*> args,
        rpy::Slice<rpy::dimn_t> arg_sizes,
        const rpy::device::KernelLaunchParams& params
) const
{
    if (is_cl_kernel(content)) {
        return cl::kernel_interface()->launch_kernel_async(
                cl_kernel(content), queue, args, arg_sizes, params
        );
    }
}

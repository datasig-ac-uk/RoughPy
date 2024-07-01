//
// Created by sam on 01/07/24.
//

#ifndef STANDARD_KERNEL_ARGUMENTS_H
#define STANDARD_KERNEL_ARGUMENTS_H

#include "kernel.h"

#include <roughpy/core/container/vector.h>

namespace rpy {
namespace devices {


struct KernelArg
{
    void* ptr;
};

class StandardKernelArguments : public KernelArguments {
    containers::SmallVec<KernelArg, 4> m_args;
    containers::SmallVec<TypePtr, 1> m_types;
    containers::SmallVec<KernelArgumentType, 4> m_signature;

    void add_type(TypePtr tp) noexcept
    {
        if (!ranges::contains(m_types, tp)) {
            m_types.push_back(std::move(tp));
        }
    }

public:

    explicit StandardKernelArguments(containers::SmallVec<KernelArgumentType, 4> signature)
        : m_signature(std::move(signature))
    {
        m_args.reserve(m_signature.size());
    }

    ~StandardKernelArguments() override;



    containers::Vec<void*> raw_pointers() const override;
    Device get_device() const noexcept override;
    Slice<const TypePtr> get_types() const noexcept override;
    void bind(Buffer buffer) override;
    void bind(ConstReference value) override;
    void bind(const operators::Operator& op) override;
    void update_arg(dimn_t idx, Buffer buffer) override;
    void update_arg(dimn_t idx, ConstReference value) override;
    void update_arg(dimn_t idx, const operators::Operator& op) override;
    dimn_t num_args() const noexcept override;
    dimn_t true_num_args() const noexcept override;
    dimn_t num_bound_args() const noexcept override;
};

} // devices
} // rpy

#endif //STANDARD_KERNEL_ARGUMENTS_H

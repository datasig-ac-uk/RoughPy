//
// Created by sam on 01/07/24.
//

#ifndef STANDARD_KERNEL_ARGUMENTS_H
#define STANDARD_KERNEL_ARGUMENTS_H

#include "kernel.h"

#include <roughpy/core/container/vector.h>

namespace rpy {
namespace devices {

class StandardKernelArguments : public KernelArguments
{

    struct KernelArg {
        void* ptr;
    };

    containers::SmallVec<KernelArg, 2> m_args;
    containers::SmallVec<TypePtr, 1> m_types;
    containers::SmallVec<const Type*, 1> m_generics;

    using Param = typename KernelSignature::Parameter;

    containers::SmallVec<Param, 4> m_signature;

    void add_type(TypePtr tp) noexcept
    {
        if (!ranges::contains(m_types, tp)) {
            m_types.push_back(std::move(tp));
        }
    }

    void check_or_set_type(dimn_t index, const TypePtr& type);

public:
    explicit StandardKernelArguments(const KernelSignature& signature);

    ~StandardKernelArguments() override;

    containers::Vec<void*> raw_pointers() const override;
    Device get_device() const noexcept override;
    Slice<const TypePtr> get_types() const noexcept override;

    Slice<const Type* const>
    get_generic_types() const noexcept override;
    containers::SmallVec<dimn_t, 2> get_sizes() const noexcept override;

    void bind(Buffer buffer) override;
    void bind(ConstReference value) override;
    void bind(const operators::Operator& op) override;
    void update_arg(dimn_t idx, Buffer buffer) override;
    void update_arg(dimn_t idx, ConstReference value) override;
    void update_arg(dimn_t idx, const operators::Operator& op) override;
    dimn_t num_args() const noexcept override;
    dimn_t true_num_args() const noexcept override;
    dimn_t num_bound_args() const noexcept override;

    const Type* get_type(dimn_t index) const override;
    void* get_raw_ptr(dimn_t index) const override;
};

}// namespace devices
}// namespace rpy

#endif// STANDARD_KERNEL_ARGUMENTS_H

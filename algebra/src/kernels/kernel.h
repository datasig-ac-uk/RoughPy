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

#include "key_algorithms.h"
#include "mutable_vector_element.h"
#include "vector.h"

namespace rpy {
namespace algebra {

namespace dtl {

using KArg = devices::KernelArgument;

template <typename V>
struct VectorArg {
    V* data;
    devices::Buffer mapped_scalars;
};

struct MutableDenseVectorArg : VectorArg<VectorData> {
};

struct MutableSparseVectorArg : VectorArg<VectorData> {
    devices::Buffer mapped_keys;
};

struct ConstDenseVectorArg : VectorArg<const VectorData> {
};
struct ConstSparseVectorArg : VectorArg<const VectorData> {
    devices::Buffer mapped_keys;
};

struct MutableScalarArg {
    scalars::Scalar* arg;
};

struct ConstScalarArg {
    const scalars::Scalar* arg;
};

template <typename Arg>
struct ArgData {
    Arg m_data;

    Arg* operator->() noexcept { return &m_data; }
    Arg& operator*() noexcept { return m_data; }
};

template <>
struct ArgData<VectorData> {
    VectorData* p_data;

    VectorData* operator->() noexcept { return p_data; }
    VectorData& operator*() noexcept { return *p_data; }
};

template <>
struct ArgData<const VectorData> {
    const VectorData* p_data;

    const VectorData* operator->() noexcept { return p_data; }
    VectorData& operator*() noexcept { return *p_data; }
};

struct MutableVectorArg {
    using arg_type = VectorData&;
    using data = ArgData<VectorData>;
};
struct ConstVectorArg {
    using arg_type = const VectorData&;
    using data = ArgData<const VectorData>;
};
struct ConstScalarArg {
    using arg_type = const scalars::Scalar&;
    using data = ArgData<const scalars::Scalar>;
};
struct MutableScalarArg {
    using arg_type = scalars::Scalar&;
    using data = ArgData<scalars::Scalar>;
};

class KernelLaunchData
{
    devices::KernelLaunchParams m_params;
    string m_suffix;
    string_view m_name;
};

template <typename F, typename... Ts>
auto curry(F&& func, Ts&&... curried_args)
{
    return [=](auto... more_args) {
        return func(std::forward<Ts>(curried_args)..., more_args...);
    };
}

template <typename... ArgSpec>
class BoundArgs;

template <typename FirstArgSpec, typename... RemainingArgSpec>
class BoundArgs<FirstArgSpec, RemainingArgSpec...>
    : BoundArgs<RemainingArgSpec...>, FirstArgSpec::data
{
    using base_t = BoundArgs<RemainingArgSpec...>;

    template <typename F, typename... Args>
    decltype(auto) eval(F&& func, Args&&... args)
    {
        return base_t::eval(std::forward<F>(func), std::forward<Args>(args)...);
    }
};

template <>
class BoundArgs<>
{
};

}// namespace dtl
template <typename Derived, typename... ArgSpec>
class VectorKernelBase
{

    template <typename Spec>
    using arg_type = typename Spec::arg_type;

protected:
    Derived& instance() noexcept { return static_cast<Derived&>(*this); }
    const Derived& instance() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }

public:
    decltype(auto) operator()(arg_type<ArgSpec>... spec) const { return 0; }
};

}// namespace algebra
}// namespace rpy

#endif// KERNEL_H

// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// ReSharper disable CppClassCanBeFinal
#ifndef ROUGHPY_DEVICE_KERNEL_H_
#define ROUGHPY_DEVICE_KERNEL_H_

#include "core.h"

#include <roughpy/core/container/vector.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/slice.h>
#include <roughpy/core/sync.h>
#include <roughpy/core/types.h>

#include <limits>

#include "buffer.h"
#include "device_object_base.h"
#include "event.h"
#include "kernel_operators.h"
#include "kernel_parameters.h"
#include "value.h"

namespace rpy {
namespace devices {

#ifndef ROUGHPY_DEVICE_SUPPORT_OPERATORS_H
namespace operators {

template <typename T>
struct Identity;

template <typename T>
struct Uminus;

template <typename T>
struct Add;

template <typename T>
struct Sub;

template <typename T>
struct LeftScalarMultiply;

template <typename T>
struct RightScalarMultiply;

template <typename T, typename S>
struct RightScalarDivide;

template <typename T>
struct FusedLeftScalarMultiplyAdd;

template <typename T>
struct FusedRightScalarMultiplyAdd;

template <typename T>
struct FusedLeftScalarMultiplySub;

template <typename T>
struct FusedRightScalarMultiplySub;

template <typename T, typename S>
struct FusedRightScalarDivideAdd;

template <typename T, typename S>
struct FusedRightScalarDivideSub;

}// namespace operators
#endif

/**
 * @class KernelLaunchParams
 * @brief Class representing the launch parameters for a kernel.
 */
class ROUGHPY_DEVICES_EXPORT KernelLaunchParams
{
    Size3 m_work_dims{1, 1, 1};
    Size3 m_group_size{1, 1, 1};
    optional<Dim3> m_offsets{};

public:
    explicit KernelLaunchParams(Size3 work_dims) : m_work_dims(work_dims) {}

    explicit KernelLaunchParams(Size3 work_dims, Size3 group_size)
        : m_work_dims(work_dims),
          m_group_size(group_size)
    {}

    RPY_NO_DISCARD bool has_work() const noexcept;

    RPY_NO_DISCARD Size3 work_dims() const noexcept { return m_work_dims; }
    RPY_NO_DISCARD Size3 work_groups() const noexcept { return m_group_size; }

    RPY_NO_DISCARD dimn_t total_work_size() const noexcept;

    RPY_NO_DISCARD dsize_t num_dims() const noexcept;

    RPY_NO_DISCARD Size3 num_work_groups() const noexcept;

    RPY_NO_DISCARD Size3 underflow_of_groups() const noexcept;

    KernelLaunchParams();
};

class KernelArguments;

class ROUGHPY_DEVICES_EXPORT KernelSignature : public platform::SmallObjectBase
{
public:
    struct Parameter {
        uint8_t type_index;

        /**
         *  @brief Kind of the parameter
         *
         *  We don't use the params::ParameterType enumerator here because we
         *  want to leave open the possibility that the signature contains
         *  parameter types that are not one of those listed above. However, for
         *  basic kernels, the signature will only ever contain one of the
         *  values in the enumerator.
         */
        uint8_t param_kind;
    };

    virtual ~KernelSignature();

    RPY_NO_DISCARD virtual std::unique_ptr<KernelArguments> new_binding() const;

    RPY_NO_DISCARD virtual Slice<const Parameter> parameters() const noexcept
            = 0;

    RPY_NO_DISCARD virtual dimn_t num_parameters() const noexcept = 0;

    RPY_NO_DISCARD virtual Slice<const TypePtr> types() const noexcept = 0;
};

namespace dtl {
template <typename T, dimn_t N>
inline constexpr bool store_inline = N * sizeof(T) <= sizeof(void*);

template <typename T>
struct GetType {
    static TypePtr get_type() { return devices::get_type<T>(); }
};

template <int N>
struct GetType<params::GenericParam<N>> {
    static TypePtr get_type() { return nullptr; }
};

template <>
struct GetType<Value> {
    static TypePtr get_type() { return nullptr; }
};

template <typename T, typename... Ts>
constexpr bool contains_type() noexcept
{
    return (... || is_same_v<T, Ts>);
}

template <typename... Ts>
struct ParamTypeList {

    template <typename T>
    using push_back = conditional_t<
            contains_type<T, Ts...>(),
            ParamTypeList,
            ParamTypeList<Ts..., T>>;

    static constexpr auto size = sizeof...(Ts);
};

template <typename Q, typename T, typename... Ts>
constexpr dimn_t index_of_type(ParamTypeList<T, Ts...>) noexcept
{
    return is_same_v<Q, T> ? 0 : (index_of_type<Q>(ParamTypeList<Ts...>()) + 1);
}

template <typename Q, typename T>
constexpr dimn_t index_of_type(ParamTypeList<T>) noexcept
{
    static_assert(
            is_same_v<Q, T>,
            "The query type does not appear in the list"
    );
    return 0;
}

template <typename... Ts>
constexpr dimn_t param_list_size(ParamTypeList<Ts...>) noexcept
{
    return sizeof...(Ts);
}

template <typename T, dimn_t N, bool = store_inline<T, N>>
struct StaticInlineArray {
    T values[N]{};
};

template <typename T, dimn_t N>
struct StaticInlineArray<T, N, false> {
    T* values;

    StaticInlineArray()
        : values(static_cast<T*>(platform::alloc_small(sizeof(T) * N)))
    {}

    ~StaticInlineArray() { platform::free_small(values, sizeof(T) * N); }
};

template <typename... ArgSpecs>
struct SignatureHelper;

template <typename, typename = void>
struct TypeOfArgImpl {
    using type = void;
};

template <typename T>
struct TypeOfArgImpl<T, void_t<typename T::bind_type>> {
    using type = typename T::bind_type;
};

template <typename T>
using TypeOfArg = typename T::bind_type;

template <typename T>
struct DoNoInit {
    template <typename TypeList>
    static constexpr typename KernelSignature::Parameter*
    init_param(TypeList, typename KernelSignature::Parameter* ptr)
    {
        return ptr;
    }
};

template <typename T>
struct DoInit {
    template <typename TypeList>
    static constexpr typename KernelSignature::Parameter*
    init_param(TypeList type_list, typename KernelSignature::Parameter* ptr)
    {
        static_assert(
                index_of_type<TypeOfArg<T>>(type_list)
                        <= std::numeric_limits<uint8_t>::max(),
                "the number of arguments exceeds the maximum size of the "
                "storage type"
        );
        construct_inplace(
                ptr,
                static_cast<uint8_t>(index_of_type<TypeOfArg<T>>(type_list)),
                static_cast<uint8_t>(T::kind)
        );
        return ++ptr;
    }
};

template <typename T>
using InitParam
        = conditional_t<params::is_parameter<T>, DoInit<T>, DoNoInit<T>>;

template <typename ArgSpec, typename... ArgSpecs>
struct SignatureHelper<ArgSpec, ArgSpecs...> {
    using next_t = SignatureHelper<ArgSpecs...>;

    using Parameter = typename KernelSignature::Parameter;
    using type_list =
            typename next_t::type_list::template push_back<TypeOfArg<ArgSpec>>;
    using param_list = typename next_t::param_list::template push_back<ArgSpec>;

    // The ArgSpec might not contribute any params to the list
    static constexpr dimn_t num_params
            = next_t::num_params + (params::is_parameter<ArgSpec> ? 1 : 0);

    template <typename... Ts>
    static constexpr void
    init_params(ParamTypeList<Ts...> list, Parameter* ptr) noexcept
    {
        next_t::init_params(
                std::move(list),
                InitParam<ArgSpec>::init_param(list, ptr)
        );
    }
};

template <>
struct SignatureHelper<> {
    using Parameter = typename KernelSignature::Parameter;
    static constexpr dimn_t num_params = 0;
    using type_list = ParamTypeList<>;
    using param_list = params::ParamList<>;

    template <typename... Ts>
    static constexpr void
    init_params(ParamTypeList<Ts...>, Parameter* RPY_UNUSED_VAR ptr) noexcept
    {
        // We've finished initialising now.
    }
};

template <typename T>
inline void init_types(ParamTypeList<T>, TypePtr* ptr)
{
    construct_inplace(ptr, GetType<T>::get_type());
}

template <typename T, typename... Ts>
inline void init_types(ParamTypeList<T, Ts...>, TypePtr* ptr)
{
    construct_inplace(ptr, GetType<T>::get_type());
    init_types(ParamTypeList<Ts...>{}, ++ptr);
}

}// namespace dtl

template <typename... ArgSpec>
class StandardKernelSignature : public KernelSignature
{
    using helper_t = dtl::SignatureHelper<ArgSpec...>;

public:
    using ParamsList = typename helper_t::param_list;
    using ParamTypeList = typename helper_t::type_list;

    static constexpr auto num_params = ParamsList::size;

    dtl::StaticInlineArray<Parameter, helper_t::num_params> m_parameters;
    dtl::StaticInlineArray<TypePtr, param_list_size(ParamTypeList())> m_types;

    StandardKernelSignature()
    {
        init_types(ParamTypeList{}, m_types.values);
        helper_t::init_params(ParamTypeList(), m_parameters.values);
    }

    /// The standard signature uses the default argument binding.
    using KernelSignature::new_binding;

    RPY_NO_DISCARD Slice<const Parameter> parameters() const noexcept override;
    RPY_NO_DISCARD Slice<const TypePtr> types() const noexcept override;

    RPY_NO_DISCARD dimn_t num_parameters() const noexcept override
    {
        return num_params;
    }
    static const StandardKernelSignature* make() noexcept;
};

/**
 * @class KernelArguments
 * @brief Class representing the bound arguments for a kernel.
 */
class ROUGHPY_DEVICES_EXPORT KernelArguments : public platform::SmallObjectBase
{
public:
    virtual ~KernelArguments();

    virtual containers::Vec<void*> raw_pointers() const = 0;
    virtual Device get_device() const noexcept = 0;
    virtual Slice<const TypePtr> get_types() const noexcept = 0;
    virtual Slice<const Type* const> get_generic_types() const noexcept = 0;

    virtual containers::SmallVec<dimn_t, 2> get_sizes() const noexcept = 0;

    virtual void bind(Buffer buffer) = 0;
    virtual void bind(ConstReference value) = 0;
    virtual void bind(const operators::Operator& op) = 0;

    virtual void update_arg(dimn_t idx, Buffer buffer) = 0;
    virtual void update_arg(dimn_t idx, ConstReference value) = 0;
    virtual void update_arg(dimn_t idx, const operators::Operator& op) = 0;

    virtual dimn_t num_args() const noexcept = 0;
    virtual dimn_t true_num_args() const noexcept;
    virtual dimn_t num_bound_args() const noexcept = 0;

    virtual const Type* get_type(dimn_t index) const = 0;
    virtual void* get_raw_ptr(dimn_t index) const = 0;
};

/*
 * Now comes the most difficult part of the argument binding process, unpacking
 * the bound arguments to call the C++ function that defines the kernel. This
 * might not be necessary for all kernels (CUDA kernels for instance will be
 * invoked by passing arg pointers as void** to the driver), but in most cases
 * the kernel will wrap a standard function. To make this process more smooth,
 * we set up some mechanisms to automate wherever possible.
 */

template <typename T>
struct ElementType {
    using type = T;
};

template <typename T>
struct ElementType<Slice<T>> {
    using type = T;
};

template <typename T>
using element_type = typename ElementType<T>::type;

template <typename T, typename = void>
inline constexpr bool is_callable = false;

template <typename T>
inline constexpr bool is_callable<T, void_t<return_type_t<T>>> = true;

template <typename BoundType, typename ArgType, typename SFINAE = void>
struct ArgumentDecoder;

template <typename T>
struct ArgumentDecoder<params::Buffer<T>, const Buffer&> {
    static const Buffer& decode(const void* arg)
    {
        return *static_cast<const Buffer*>(arg);
    }
};

template <typename T>
struct ArgumentDecoder<
        params::Buffer<T>,
        Slice<const T>,
        enable_if_t<!is_same_v<T, Value>>> {
    static Slice<const T> decode(const void* arg)
    {
        return static_cast<const Buffer*>(arg)->as_slice<T>();
    }
};

template <>
struct ArgumentDecoder<params::Buffer<Value>, Slice<const Value>> {
    static Slice<const Value> decode(const void* arg)
    {
        return static_cast<const Buffer*>(arg)->as_value_slice();
    }
};

template <int N>
struct ArgumentDecoder<
        params::Buffer<params::GenericParam<N>>,
        Slice<const Value>> {
    static Slice<const Value> decode(const void* arg)
    {
        return static_cast<const Buffer*>(arg)->as_value_slice();
    }
};

template <int N, typename T>
struct ArgumentDecoder<
        params::Buffer<params::GenericParam<N>>,
        Slice<const T>,
        enable_if_t<!is_same_v<T, Value>>> {
    static Slice<const T> decode(const void* arg)
    {
        return static_cast<const Buffer*>(arg)->as_slice<T>();
    }
};

template <typename T>
struct ArgumentDecoder<params::ResultBuffer<T>, Buffer&> {
    static Buffer& decode(void* arg) { return *static_cast<Buffer*>(arg); }
};

template <typename T>
struct ArgumentDecoder<
        params::ResultBuffer<T>,
        Slice<T>,
        enable_if_t<!is_same_v<T, Value>>> {
    static Slice<T> decode(void* arg)
    {
        return static_cast<Buffer*>(arg)->as_mut_slice<T>();
    }
};

template <>
struct ArgumentDecoder<params::ResultBuffer<Value>, Slice<Value>> {
    static Slice<Value> decode(void* arg)
    {
        return static_cast<Buffer*>(arg)->as_mut_value_slice();
    }
};

template <int N>
struct ArgumentDecoder<
        params::ResultBuffer<params::GenericParam<N>>,
        Slice<Value>> {
    static Slice<Value> decode(void* arg)
    {
        return static_cast<Buffer*>(arg)->as_mut_value_slice();
    }
};

template <int N, typename T>
struct ArgumentDecoder<
        params::ResultBuffer<params::GenericParam<N>>,
        Slice<T>,
        enable_if_t<!is_same_v<T, Value>>> {
    static Slice<const T> decode(void* arg)
    {
        return static_cast<const Buffer*>(arg)->as_mut_slice<T>();
    }
};

template <typename T>
struct ArgumentDecoder<params::Value<T>, ConstReference> {
    static ConstReference decode(const void* arg)
    {
        return ConstReference(get_type<T>(), arg);
    }
};

template <typename T>
struct ArgumentDecoder<params::Value<T>, T> {
    static T decode(void* arg) { return *static_cast<const T*>(arg); }
};

template <int N, typename T>
struct ArgumentDecoder<params::Value<params::GenericParam<N>>, T> {
    static T decode(void* arg) { return *static_cast<const T*>(arg); }
};

template <typename T>
struct ArgumentDecoder<params::Operator<T>, const operators::Operator&> {
    static const operators::Operator& decode(const void* arg)
    {
        return *static_cast<const operators::Operator*>(arg);
    }
};

template <typename T>
struct ArgumentDecoder<params::Operator<T>, operators::Identity<T>> {
    static operators::Identity<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::Identity);
        return operators::Identity<T>();
    }
};

template <typename T>
struct ArgumentDecoder<params::Operator<T>, operators::Uminus<T>> {
    static operators::Uminus<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::UnaryMinus);
        return operators::Uminus<T>();
    }
};

template <typename T>
struct ArgumentDecoder<params::Operator<T>, operators::LeftScalarMultiply<T>> {
    static operators::LeftScalarMultiply<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::LeftMultiply);
        const auto& data
                = operators::op_cast<operators::LeftMultiplyOperator>(op).data(
                );
        return operators::LeftScalarMultiply<T>(value_cast<T>(data));
    }
};

template <typename T>
struct ArgumentDecoder<params::Operator<T>, operators::RightScalarMultiply<T>> {
    static operators::RightScalarMultiply<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::RightMultiply);
        const auto& data
                = operators::op_cast<operators::RightMultiplyOperator>(op).data(
                );
        return operators::RightScalarMultiply<T>(value_cast<T>(data));
    }
};

template <typename T>
struct ArgumentDecoder<params::Operator<T>, operators::Add<T>> {
    static operators::Add<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::Addition);
        return operators::Add<T>();
    }
};

template <typename T>
struct ArgumentDecoder<params::Operator<T>, operators::Sub<T>> {
    static operators::Sub<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::Subtraction);
        return operators::Sub<T>();
    }
};

template <typename T>
struct ArgumentDecoder<
        params::Operator<T>,
        operators::FusedLeftScalarMultiplyAdd<T>> {
    static operators::FusedLeftScalarMultiplyAdd<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::FusedLeftMultiplyAdd);
        const auto& data
                = operators::op_cast<operators::FusedLeftMultiplyAddOperator>(op
                )
                          .data();
        return operators::FusedLeftScalarMultiplyAdd<T>(value_cast<T>(data));
    }
};

template <typename T>
struct ArgumentDecoder<
        params::Operator<T>,
        operators::FusedRightScalarMultiplyAdd<T>> {
    static operators::FusedRightScalarMultiplyAdd<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::FusedRightMultiplyAdd);
        const auto& data
                = operators::op_cast<operators::FusedRightMultiplyAddOperator>(
                          op
                )
                          .data();
        return operators::FusedRightScalarMultiplyAdd<T>(value_cast<T>(data));
    }
};

template <typename T>
struct ArgumentDecoder<
        params::Operator<T>,
        operators::FusedLeftScalarMultiplySub<T>> {
    static operators::FusedLeftScalarMultiplySub<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::FusedLeftMultiplySub);
        const auto& data
                = operators::op_cast<operators::FusedLeftMultiplySubOperator>(op
                )
                          .data();
        return operators::FusedLeftScalarMultiplySub<T>(value_cast<T>(data));
    }
};

template <typename T>
struct ArgumentDecoder<
        params::Operator<T>,
        operators::FusedRightScalarMultiplySub<T>> {
    static operators::FusedRightScalarMultiplySub<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::FusedRightMultiplySub);
        const auto& data
                = operators::op_cast<operators::FusedRightMultiplySubOperator>(
                          op
                )
                          .data();
        return operators::FusedRightScalarMultiplySub<T>(value_cast<T>(data));
    }
};

template <int N, typename T>
struct ArgumentDecoder<
        params::Operator<params::GenericParam<N>>,
        operators::Identity<T>> {
    static operators::Identity<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::Identity);
        return operators::Identity<T>();
    }
};

template <int N, typename T>
struct ArgumentDecoder<
        params::Operator<params::GenericParam<N>>,
        operators::Uminus<T>> {
    static operators::Uminus<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::UnaryMinus);
        return operators::Uminus<T>();
    }
};

template <int N, typename T>
struct ArgumentDecoder<
        params::Operator<params::GenericParam<N>>,
        operators::LeftScalarMultiply<T>> {
    static operators::LeftScalarMultiply<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::LeftMultiply);
        const auto& data
                = operators::op_cast<operators::LeftMultiplyOperator>(op).data(
                );
        return operators::LeftScalarMultiply<T>(value_cast<T>(data));
    }
};

template <int N, typename T>
struct ArgumentDecoder<
        params::Operator<params::GenericParam<N>>,
        operators::RightScalarMultiply<T>> {
    static operators::RightScalarMultiply<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::RightMultiply);
        const auto& data
                = operators::op_cast<operators::RightMultiplyOperator>(op).data(
                );
        return operators::RightScalarMultiply<T>(value_cast<T>(data));
    }
};

template <int N, typename T>
struct ArgumentDecoder<
        params::Operator<params::GenericParam<N>>,
        operators::Add<T>> {
    static operators::Add<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::Addition);
        return operators::Add<T>();
    }
};

template <int N, typename T>
struct ArgumentDecoder<
        params::Operator<params::GenericParam<N>>,
        operators::Sub<T>> {
    static operators::Sub<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::Subtraction);
        return operators::Sub<T>();
    }
};

template <int N, typename T>
struct ArgumentDecoder<
        params::Operator<params::GenericParam<N>>,
        operators::FusedLeftScalarMultiplyAdd<T>> {
    static operators::FusedLeftScalarMultiplyAdd<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::FusedLeftMultiplyAdd);
        const auto& data
                = operators::op_cast<operators::FusedLeftMultiplyAddOperator>(op
                )
                          .data();
        return operators::FusedLeftScalarMultiplyAdd<T>(value_cast<T>(data));
    }
};

template <int N, typename T>
struct ArgumentDecoder<
        params::Operator<params::GenericParam<N>>,
        operators::FusedRightScalarMultiplyAdd<T>> {
    static operators::FusedRightScalarMultiplyAdd<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::FusedRightMultiplyAdd);
        const auto& data
                = operators::op_cast<operators::FusedRightMultiplyAddOperator>(
                          op
                )
                          .data();
        return operators::FusedRightScalarMultiplyAdd<T>(value_cast<T>(data));
    }
};

template <int N, typename T>
struct ArgumentDecoder<
        params::Operator<params::GenericParam<N>>,
        operators::FusedLeftScalarMultiplySub<T>> {
    static operators::FusedLeftScalarMultiplySub<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::FusedLeftMultiplySub);
        const auto& data
                = operators::op_cast<operators::FusedLeftMultiplySubOperator>(op
                )
                          .data();
        return operators::FusedLeftScalarMultiplySub<T>(value_cast<T>(data));
    }
};

template <int N, typename T>
struct ArgumentDecoder<
        params::Operator<params::GenericParam<N>>,
        operators::FusedRightScalarMultiplySub<T>> {
    static operators::FusedRightScalarMultiplySub<T> decode(const void* arg)
    {
        const auto& op = *static_cast<const operators::Operator*>(arg);
        RPY_CHECK(op.kind() == operators::Operator::FusedRightMultiplySub);
        const auto& data
                = operators::op_cast<operators::FusedRightMultiplySubOperator>(
                          op
                )
                          .data();
        return operators::FusedRightScalarMultiplySub<T>(value_cast<T>(data));
    }
};
namespace dtl {

template <typename List, typename ArgTuple>
class ArgumentBinder;

template <typename Param, typename Arg>
class ArgumentBinder<params::ParamList<Param>, params::ParamList<Arg>>
{
    using Decoder = ArgumentDecoder<Param, decay_t<Arg>>;

    template <typename PL, typename AL>
    friend class ArgumentBinder;

    template <typename F>
    static decltype(auto)
    eval_impl(F&& fn, const KernelArguments& args, dimn_t idx)
    {
        return fn(Decoder::decode(args.get_raw_ptr(idx)));
    }

public:
    template <typename F>
    static decltype(auto) eval(F&& fn, const KernelArguments& args)
    {
        return eval_impl(std::forward<F>(fn), args, 0);
    }
};

template <typename Param, typename... Params, typename Arg, typename... Args>
class ArgumentBinder<
        params::ParamList<Param, Params...>,
        params::ParamList<Arg, Args...>>
{
    using params_list_t = params::ParamList<Param, Params...>;
    using args_list_t = params::ParamList<Arg, Args...>;

    static_assert(
            params_list_t::size == args_list_t::size,
            "mismatch between number of args and number of parameters"
    );

    using Decoder = ArgumentDecoder<Param, decay_t<Arg>>;
    using next_t = ArgumentBinder<
            params::ParamList<Params...>,
            params::ParamList<Args...>>;

    template <typename PL, typename AL>
    friend class ArgumentBinder;

    template <typename F>
    static decltype(auto)
    eval_impl(F&& fn, const KernelArguments& args, dimn_t idx)
    {
        auto* arg_ptr = args.get_raw_ptr(idx);

        return next_t::template eval_impl(
                [f = std::forward<F>(fn),
                 value = Decoder::decode(arg_ptr)](auto&&... remaining) {
                    return f(
                            value,
                            std::forward<decltype(remaining)>(remaining)...
                    );
                },
                args,
                ++idx
        );
    }

public:
    template <typename F>
    static decltype(auto) eval(F&& fn, const KernelArguments& args)
    {
        return eval_impl(std::forward<F>(fn), args, 0);
    }
};

template <typename L>
struct KernelParamTypesImpl {
    using type = L;
};

template <typename... Ts>
struct KernelParamTypesImpl<
        params::ParamList<const KernelLaunchParams&, Ts...>> {
    using type = params::ParamList<Ts...>;
};

template <typename F>
using kernel_fn_args =
        typename KernelParamTypesImpl<args_t<F, params::ParamList>>::type;

}// namespace dtl

template <typename ParamList, typename F>
using ArgumentBinder = dtl::ArgumentBinder<ParamList, dtl::kernel_fn_args<F>>;

class ROUGHPY_DEVICES_EXPORT KernelSpec
{
    string m_name;
    const KernelArguments* p_kernel_args;

public:
    explicit KernelSpec(string name, const KernelArguments& kargs)
        : m_name(std::move(name)),
          p_kernel_args(&kargs)
    {}

    RPY_NO_DISCARD string_view name() const noexcept { return m_name; }

    RPY_NO_DISCARD Device get_device() const noexcept;
    RPY_NO_DISCARD Slice<const TypePtr> get_types() const noexcept;
};

/**
 * @class KernelInterface
 * @brief Interface representing a kernel for execution on a platform.
 *
 * This interface provides methods to get the name of the kernel, the number of
 * arguments, and launching the kernel asynchronously or synchronously with
 * given arguments.
 */
class ROUGHPY_DEVICES_EXPORT KernelInterface : public dtl::InterfaceBase
{
    const KernelSignature* p_signature;

public:
    using object_t = Kernel;

    explicit KernelInterface(const KernelSignature* signature)
        : p_signature(signature)
    {
        RPY_DBG_ASSERT(p_signature != nullptr);
    }

    /**
     * @brief Returns the name of the kernel.
     *
     * @return The name of the kernel.
     */
    RPY_NO_DISCARD virtual string name() const;

    /**
     * @brief Get the number of arguments required by the kernel.
     *
     * This method is a virtual member function of the KernelInterface class.
     * It returns the number of arguments required by the kernel.
     *
     * @return The number of arguments required by the kernel.
     */
    RPY_NO_DISCARD virtual dimn_t num_args() const;

    /**
     * @brief Returns a constant reference to the kernel signature.
     *
     * This method returns a constant reference to the kernel signature. The
     * kernel signature represents the launch parameters for a kernel.
     *
     * @return A constant reference to the kernel signature.
     */
    RPY_NO_DISCARD const KernelSignature& signature() const noexcept
    {
        return *p_signature;
    };

    /**
     * @brief Asynchronously launches a kernel on a specified queue.
     *
     * This method asynchronously launches a kernel on the specified queue with
     * the given launch parameters and the provided kernel arguments.
     *
     * @param queue The queue on which to launch the kernel.
     * @param params The launch parameters for the kernel.
     * @param args The kernel arguments.
     * @return An event representing the asynchronous kernel launch.
     */
    RPY_NO_DISCARD virtual Event launch_kernel_async(
            Queue& queue,
            const KernelLaunchParams& params,
            const KernelArguments& args
    ) const;

    /**
     * @brief Synchronously launches a kernel on a specified queue.
     *
     * This method synchronously launches a kernel on the specified queue with
     * the given launch parameters and the provided kernel arguments.
     *
     * @param queue The queue on which to launch the kernel.
     * @param params The launch parameters for the kernel.
     * @param args The kernel arguments.
     * @return The status of the event representing the synchronous kernel
     * launch.
     */
    virtual EventStatus launch_kernel_sync(
            Queue& queue,
            const KernelLaunchParams& params,
            const KernelArguments& args
    ) const;
};

#ifdef RPY_PLATFORM_WINDOWS
#  ifdef RoughPy_Platform_EXPORTS
namespace dtl {
extern template class ObjectBase<KernelInterface, Kernel>;
}
#  else
namespace dtl {
template class RPY_DLL_IMPORT ObjectBase<KernelInterface, Kernel>;
}
#  endif
#else
namespace dtl {
extern template class ROUGHPY_DEVICES_EXPORT
        ObjectBase<KernelInterface, Kernel>;
}
#endif

/**
 * @class Kernel
 * @brief Class representing a kernel.
 * @see KernelLaunchParams
 *
 * This class represents a kernel and provides methods for launching the kernel.
 * The launch methods allow the kernel to be launched asynchronously or
 * synchronously in a specified queue with specified launch parameters and
 * arguments. The operator() can be used as a shorthand for launching the kernel
 * with a variable number of arguments.
 */
class ROUGHPY_DEVICES_EXPORT Kernel
    : public dtl::ObjectBase<KernelInterface, Kernel>
{
    using base_t = dtl::ObjectBase<KernelInterface, Kernel>;

public:
    using base_t::base_t;

    /**
     * @brief Checks if the object is a no-op (null).
     *
     * @return True if the object is a no-op (null), false otherwise.
     *
     * @note The function is_noop() is a const member function that does not
     * throw any exceptions. It returns a boolean value indicating whether the
     * object is a no-op (null) or not. The function is implemented by calling
     * the is_null() function. If the object is a no-op (null), the function
     * will return true, otherwise it will return false.
     */
    RPY_NO_DISCARD bool is_nop() const noexcept { return is_null(); }

    /**
     * @brief Retrieves the name of the kernel.
     *
     * This method returns the name of the kernel represented by the Kernel
     * object. If the underlying implementation is not set (impl() returns
     * nullptr), an empty string is returned.
     *
     * @return The name of the kernel. If the underlying implementation is not
     * set, returns an empty string.
     */
    RPY_NO_DISCARD string name() const;

    /**
     * @brief Retrieves the number of arguments of the kernel.
     *
     * This method returns the number of arguments of the kernel represented by
     * the Kernel object. If the underlying implementation is not set (impl()
     * returns nullptr), 0 is returned.
     *
     * @return The number of arguments of the kernel. If the underlying
     * implementation is not set, returns 0.
     */
    RPY_NO_DISCARD dimn_t num_args() const;

    /**
     * @fn const KernelSignature& Kernel::signature() const noexcept
     * @brief Returns the signature of the kernel.
     * @details This method returns the signature of the kernel. The signature
     * represents the types of the kernel's arguments and return value.
     * @pre The kernel object must not be null.
     * @return A constant reference to the KernelSignature object representing
     * the signature of the kernel.
     * @see KernelSignature
     */
    RPY_NO_DISCARD const KernelSignature& signature() const noexcept
    {
        RPY_DBG_ASSERT(!is_null());
        return impl()->signature();
    }

    /**
     * @brief Launches a kernel asynchronously in a queue.
     *
     * This method launches a kernel asynchronously in the specified queue with
     * the provided launch parameters and kernel arguments. If the kernel or the
     * launch parameters are not set, or if the queue is not valid for this
     * kernel, an empty Event object is returned.
     *
     * @param queue The queue in which to launch the kernel.
     * @param params The launch parameters for the kernel.
     * @param args The kernel arguments.
     *
     * @return An Event object representing the kernel launch event. If the
     * kernel or the launch parameters are not set, or if the queue is not valid
     * for this kernel, an empty Event object is returned.
     */
    RPY_NO_DISCARD Event launch_async_in_queue(
            Queue& queue,
            const KernelLaunchParams& params,
            const KernelArguments& args
    ) const;

    /**
     * @brief Launches a kernel synchronously in a queue.
     *
     * This method launches a kernel synchronously in the specified queue with
     * the provided launch parameters and kernel arguments. If the kernel or the
     * launch parameters are not set, or if the queue is not valid for this
     * kernel, an empty Event object is returned.
     *
     * @param queue The queue in which to launch the kernel.
     * @param params The launch parameters for the kernel.
     * @param args The kernel arguments.
     *
     * @return An EventStatus indicating the status of the kernel launch. If the
     * kernel or the launch parameters are not set, or if the queue is not valid
     * for this kernel, the EventStatus::Error value is returned.
     *
     * @see EventStatus
     */
    RPY_NO_DISCARD EventStatus launch_sync_in_queue(
            Queue& queue,
            const KernelLaunchParams& params,
            const KernelArguments& args
    ) const;

    /**
     * @brief Launches a kernel asynchronously in a queue.
     *
     * This method launches a kernel asynchronously in the specified queue with
     * the provided launch parameters and kernel arguments. If the kernel or the
     * launch parameters are not set, or if the queue is not valid for this
     * kernel, an empty Event object is returned.
     *
     * @param params The launch parameters for the kernel.
     * @param args The kernel arguments.
     *
     * @return An Event object representing the kernel launch event. If the
     * kernel or the launch parameters are not set, or if the queue is not valid
     * for this kernel, an empty Event object is returned.
     */
    RPY_NO_DISCARD Event launch_async(
            const KernelLaunchParams& params,
            const KernelArguments& args
    ) const;

    /**
     * @brief Launches a kernel synchronously with the specified launch
     * parameters and arguments.
     *
     * This method launches a kernel synchronously using the specified launch
     * parameters and arguments. The kernel launch is performed on the default
     * queue of the device associated with this kernel.
     *
     * @param params The launch parameters for the kernel.
     * @param args The arguments to be passed to the kernel.
     *
     * @return The status of the event generated by the kernel launch. Possible
     * values are:
     *         - CompletedSuccessfully: The kernel execution completed
     * successfully.
     *         - Queued: The kernel execution is queued and waiting to be
     * executed.
     *         - Submitted: The kernel execution has been submitted to the
     * device.
     *         - Running: The kernel execution is currently running on the
     * device.
     *         - Error: There was an error during the kernel execution.
     */
    RPY_NO_DISCARD EventStatus launch_sync(
            const KernelLaunchParams& params,
            const KernelArguments& args
    ) const;

    RPY_NO_DISCARD static containers::Vec<bitmask_t>
    construct_work_mask(const KernelLaunchParams& params);

    // template <typename... Args>
    // void operator()(const KernelLaunchParams& params, Args&&... args) const;
};

template <typename... ArgSpec>
Slice<const KernelSignature::Parameter>
StandardKernelSignature<ArgSpec...>::parameters() const noexcept
{
    return {m_parameters.values, helper_t::num_params};
}
template <typename... ArgSpec>
Slice<const TypePtr> StandardKernelSignature<ArgSpec...>::types() const noexcept
{
    return {m_types.values, dtl::param_list_size(ParamTypeList())};
}

template <typename... ArgSpec>
const StandardKernelSignature<ArgSpec...>*
StandardKernelSignature<ArgSpec...>::make() noexcept
{
    static const StandardKernelSignature value;
    return &value;
}

namespace dtl {

template <typename T>
void bind_args(KernelArguments& binding, T&& arg)
{
    binding.bind(std::forward<T>(arg));
}

template <typename T, typename... Ts>
void bind_args(KernelArguments& binding, T&& first, Ts&&... remaining)
{
    binding.bind(std::forward<T>(first));
    bind_args(binding, std::forward<Ts>(remaining)...);
}

}// namespace dtl

template <typename... Args>
EventStatus launch_kernel_sync(
        Kernel kernel,
        const KernelLaunchParams& params,
        Args&&... args
)
{
    RPY_CHECK(!kernel.is_null());
    auto binding = kernel.signature().new_binding();
    dtl::bind_args(*binding, std::forward<Args>(args)...);
    return kernel.launch_sync(params, *binding);
}

template <typename... Args>
Event launch_kernel_async(
        Kernel kernel,
        Queue& queue,
        const KernelLaunchParams& params,
        Args&&... args
)
{
    RPY_CHECK(!kernel.is_null());
    auto binding = kernel.signature().new_binding();
    dtl::bind_args(*binding, std::forward<Args>(args)...);
    return kernel.launch_async_in_queue(queue, params, *binding);
}

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_KERNEL_H_

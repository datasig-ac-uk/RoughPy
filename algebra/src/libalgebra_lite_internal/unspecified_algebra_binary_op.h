// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 18/07/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_UNSPECIFIED_ALGEBRA_BINARY_OP_H_
#define ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_UNSPECIFIED_ALGEBRA_BINARY_OP_H_

#include <roughpy/algebra/algebra_fwd.h>
#include <roughpy/algebra/context_fwd.h>

namespace rpy {
namespace algebra {

template <
        template <AlgebraType, VectorType> class CasterType, typename Context,
        typename... ExpectedArgTypes>
class UnspecifiedFunctionInvoker;

template <
        template <AlgebraType, VectorType> class CasterType, typename Context,
        typename Current, typename... Remaining>
class UnspecifiedFunctionInvoker<CasterType, Context, Current, Remaining...>
    : UnspecifiedFunctionInvoker<CasterType, Context, Remaining...>
{
    using base = UnspecifiedFunctionInvoker<CasterType, Context, Remaining...>;

    template <
            AlgebraType AType, VectorType VType, typename Operation,
            typename... Previous>
    static UnspecifiedAlgebraType cast_and_continue(
            const Context* ctx, Operation&& op, Current this_arg,
            Remaining... remaining, Previous&&... previous
    )
    {
        using caster = CasterType<AType, VType>;
        return base::eval(
                ctx, std::forward<Operation>(op), remaining...,
                std::forward<Previous>(previous)...,
                caster::cast(std::forward<Current>(this_arg))
        );
    }

    template <AlgebraType AType, typename Operation, typename... Previous>
    static enable_if_t<
            Operation::template compatible<AType>::value,
            UnspecifiedAlgebraType>
    cast_vtype(
            const Context* ctx, Operation&& op, Current this_arg,
            Remaining... remaining, Previous&&... previous
    )
    {
#define RPY_SWITCH_FN(VTYPE)                                                   \
    cast_and_continue<AType, VTYPE>(                                           \
            ctx, std::forward<Operation>(op), this_arg, remaining...,               \
            std::forward<Previous>(previous)...                                     \
    )
        RPY_MAKE_VTYPE_SWITCH(this_arg->storage_type())
#undef RPY_SWITCH_FN
    }

    template <AlgebraType AType, typename Operation, typename... Previous>
    static enable_if_t<
            !Operation::template compatible<AType>::value,
            UnspecifiedAlgebraType>
    cast_vtype(
            const Context* ctx, Operation&& op, Current last_arg,
            Previous&&... previous
    )
    {
        RPY_THROW(std::runtime_error,"this operation is not defined for these "
                                 "types");
    }
    template <typename Operation, typename... Previous>
    static UnspecifiedAlgebraType cast_atype(
            const Context* ctx, Operation&& op, Current this_arg,
            Remaining... remaining, Previous&&... previous
    )
    {
#define RPY_SWITCH_FN(ATYPE)                                                   \
    cast_vtype<ATYPE>(                                                         \
            ctx, std::forward<Operation>(op), this_arg, remaining...,               \
            std::forward<Previous>(previous)...                                     \
    )
        RPY_MAKE_ALGTYPE_SWITCH(this_arg->alg_type())
#undef RPY_SWITCH_FN
    }

public:
    template <typename Operation, typename... Previous>
    static UnspecifiedAlgebraType
    eval(const Context* ctx, Operation&& op, Current this_arg,
         Remaining... remaining, Previous&&... previous)
    {
        return cast_atype(
                ctx, std::forward<Operation>(op), this_arg, remaining...,
                std::forward<Previous>(previous)...
        );
    }

    //    template <typename Operation>
    //    static UnspecifiedAlgebraType
    //    eval(const Context* ctx, Operation&& op, Current&& this_arg,
    //         Remaining&&... remaining)
    //    {
    //        return cast_atype(
    //                ctx,
    //                std::forward<Operation>(op),
    //                std::forward<Current>(this_arg),
    //                std::forward<Remaining>(remaining)...
    //        );
    //    }
};

template <
        template <AlgebraType, VectorType> class CasterType, typename Context,
        typename Last>
class UnspecifiedFunctionInvoker<CasterType, Context, Last>
{

    template <typename Operation, typename... Args>
    static UnspecifiedAlgebraType
    eval_func(const Context* ctx, Operation&& op, Args&&... args)
    {
        using out_type = decltype(op(std::forward<Args>(args)...));
        using impl_t =
                typename dtl::alg_details_of<out_type>::implementation_type;
        return UnspecifiedAlgebraType(
                new impl_t(ctx, op(std::forward<Args>(args)...))
        );
    }

    template <
            AlgebraType AType, VectorType VType, typename Operation,
            typename... Previous>
    static UnspecifiedAlgebraType cast_and_continue(
            const Context* ctx, Operation&& op, Last this_arg,
            Previous&&... previous
    )
    {
        using caster = CasterType<AType, VType>;
        return eval_func(
                ctx, std::forward<Operation>(op), std::forward<Previous>(previous)...,
                caster::cast(this_arg)
        );
    }

    template <AlgebraType AType, typename Operation, typename... Previous>
    static enable_if_t<
            Operation::template compatible<AType>::value,
            UnspecifiedAlgebraType>
    cast_vtype(
            const Context* ctx, Operation&& op, Last this_arg,
            Previous&&... previous
    )
    {
#define RPY_SWITCH_FN(VTYPE)                                                   \
    cast_and_continue<AType, VTYPE>(                                           \
            ctx, std::forward<Operation>(op), this_arg,                             \
            std::forward<Previous>(previous)...                                     \
    )
        RPY_MAKE_VTYPE_SWITCH(this_arg->storage_type())
#undef RPY_SWITCH_FN
    }

    template <AlgebraType AType, typename Operation, typename... Previous>
    static enable_if_t<
            !Operation::template compatible<AType>::value,
            UnspecifiedAlgebraType>
    cast_vtype(
            const Context* ctx, Operation&& op, Last last_arg,
            Previous&&... previous
    )
    {
        RPY_THROW(std::runtime_error,"this operation is not defined for these "
                                 "types");
    }

    template <typename Operation, typename... Previous>
    static UnspecifiedAlgebraType cast_atype(
            const Context* ctx, Operation&& op, Last this_arg,
            Previous&&... previous
    )
    {
#define RPY_SWITCH_FN(ATYPE)                                                   \
    cast_vtype<ATYPE>(                                                         \
            ctx, std::forward<Operation>(op), this_arg,                             \
            std::forward<Previous>(previous)...                                     \
    )
        RPY_MAKE_ALGTYPE_SWITCH(this_arg->alg_type())
#undef RPY_SWITCH_FN
    }

public:
    template <typename Operation, typename... Previous>
    static UnspecifiedAlgebraType
    eval(const Context* ctx, Operation&& op, Last this_arg,
         Previous&&... previous)
    {
        return cast_atype(
                ctx, std::forward<Operation>(op), this_arg,
                std::forward<Previous>(previous)...
        );
    }
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_SRC_LIBALGEBRA_LITE_INTERNAL_UNSPECIFIED_ALGEBRA_BINARY_OP_H_

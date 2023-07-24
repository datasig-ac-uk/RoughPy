// Copyright (c) 2023 RoughPy Developers. All rights reserved.
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

#ifndef ROUGHPY_ALGEBRA_CONTEXT_H_
#define ROUGHPY_ALGEBRA_CONTEXT_H_

#include "algebra_fwd.h"
#include "context_fwd.h"

#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>

#include <roughpy/core/helpers.h>
#include <roughpy/core/macros.h>
#include <roughpy/scalars/key_scalar_array.h>
#include <roughpy/scalars/scalar_stream.h>
#include <roughpy/scalars/scalar_type.h>

#include <memory>
#include <string>
#include <vector>

#include "free_tensor_fwd.h"
#include "lie_basis.h"
#include "lie_fwd.h"
#include "shuffle_tensor_fwd.h"
#include "tensor_basis.h"

namespace rpy {
namespace algebra {

struct SignatureData {
    scalars::ScalarStream data_stream;
    std::vector<const key_type*> key_stream;
    VectorType vector_type;
};

struct DerivativeComputeInfo {
    Lie logsig_of_interval;
    Lie perturbation;
};

struct VectorConstructionData {
    scalars::KeyScalarArray data;
    VectorType vector_type = VectorType::Sparse;
};

class RPY_EXPORT ContextBase : public boost::intrusive_ref_counter<ContextBase>
{
    deg_t m_width;
    deg_t m_depth;

    MaybeOwned<const dimn_t> p_lie_sizes;
    MaybeOwned<const dimn_t> p_tensor_sizes;

protected:
    ContextBase(
            deg_t width, deg_t depth, const dimn_t* lie_sizes,
            const dimn_t* tensor_sizes
    );

public:
    virtual ~ContextBase();

    RPY_NO_DISCARD deg_t width() const noexcept { return m_width; }
    RPY_NO_DISCARD deg_t depth() const noexcept { return m_depth; }

    RPY_NO_DISCARD dimn_t lie_size(deg_t deg) const noexcept;
    RPY_NO_DISCARD dimn_t tensor_size(deg_t deg) const noexcept;
};

class RPY_EXPORT Context : public ContextBase
{
    const scalars::ScalarType* p_ctype;
    string m_ctx_backend;

protected:
    explicit Context(
            deg_t width, deg_t depth, const scalars::ScalarType* ctype,
            string&& context_backend, const dimn_t* lie_sizes = nullptr,
            const dimn_t* tensor_sizes = nullptr
    )
        : ContextBase(width, depth, lie_sizes, tensor_sizes), p_ctype(ctype),
          m_ctx_backend(std::move(context_backend))
    {}

public:
    RPY_NO_DISCARD const scalars::ScalarType* ctype() const noexcept
    {
        return p_ctype;
    }
    RPY_NO_DISCARD const string& backend() const noexcept
    {
        return m_ctx_backend;
    }

    RPY_NO_DISCARD virtual context_pointer get_alike(deg_t new_depth) const = 0;
    RPY_NO_DISCARD virtual context_pointer
    get_alike(const scalars::ScalarType* new_ctype) const
            = 0;
    RPY_NO_DISCARD virtual context_pointer
    get_alike(deg_t new_depth, const scalars::ScalarType* new_ctype) const
            = 0;
    RPY_NO_DISCARD virtual context_pointer get_alike(
            deg_t new_width, deg_t new_depth,
            const scalars::ScalarType* new_ctype
    ) const = 0;

    RPY_NO_DISCARD virtual bool check_compatible(const Context& other_ctx
    ) const noexcept;

    RPY_NO_DISCARD virtual LieBasis get_lie_basis() const = 0;
    RPY_NO_DISCARD virtual TensorBasis get_tensor_basis() const = 0;

    RPY_NO_DISCARD virtual FreeTensor
    convert(const FreeTensor& arg, optional<VectorType> new_vec_type) const
            = 0;
    RPY_NO_DISCARD virtual ShuffleTensor
    convert(const ShuffleTensor& arg, optional<VectorType> new_vec_type) const
            = 0;
    RPY_NO_DISCARD virtual Lie
    convert(const Lie& arg, optional<VectorType> new_vec_type) const
            = 0;

    RPY_NO_DISCARD virtual FreeTensor
    construct_free_tensor(const VectorConstructionData& arg) const
            = 0;
    RPY_NO_DISCARD virtual ShuffleTensor
    construct_shuffle_tensor(const VectorConstructionData& arg) const
            = 0;
    RPY_NO_DISCARD virtual Lie construct_lie(const VectorConstructionData& arg
    ) const = 0;

    RPY_NO_DISCARD virtual UnspecifiedAlgebraType
    construct(AlgebraType type, const VectorConstructionData& data) const
            = 0;

    RPY_NO_DISCARD FreeTensor zero_free_tensor(VectorType vtype) const;
    RPY_NO_DISCARD ShuffleTensor zero_shuffle_tensor(VectorType vtype) const;
    RPY_NO_DISCARD Lie zero_lie(VectorType vtype) const;

protected:
    void lie_to_tensor_fallback(FreeTensor& result, const Lie& arg) const;
    void tensor_to_lie_fallback(Lie& result, const FreeTensor& arg) const;

public:
    RPY_NO_DISCARD virtual FreeTensor lie_to_tensor(const Lie& arg) const = 0;
    RPY_NO_DISCARD virtual Lie tensor_to_lie(const FreeTensor& arg) const = 0;

protected:
    void
    cbh_fallback(FreeTensor& collector, const std::vector<Lie>& lies) const;

public:
    RPY_NO_DISCARD virtual Lie
    cbh(const std::vector<Lie>& lies, VectorType vtype) const;
    RPY_NO_DISCARD virtual Lie
    cbh(const Lie& left, const Lie& right, VectorType vtype) const;

    RPY_NO_DISCARD virtual FreeTensor to_signature(const Lie& log_signature
    ) const;
    RPY_NO_DISCARD virtual FreeTensor signature(const SignatureData& data) const
            = 0;
    RPY_NO_DISCARD virtual Lie log_signature(const SignatureData& data) const
            = 0;

    RPY_NO_DISCARD virtual FreeTensor sig_derivative(
            const std::vector<DerivativeComputeInfo>& info, VectorType vtype
    ) const = 0;

    // Functions to aid serialization
    RPY_NO_DISCARD virtual std::vector<byte>
    to_raw_bytes(AlgebraType atype, RawUnspecifiedAlgebraType alg) const;

    RPY_NO_DISCARD virtual UnspecifiedAlgebraType
    from_raw_bytes(AlgebraType atype, Slice<byte> raw_bytes) const;

    virtual UnspecifiedAlgebraType free_multiply(
            const ConstRawUnspecifiedAlgebraType left,
            const ConstRawUnspecifiedAlgebraType right
    ) const;

    virtual UnspecifiedAlgebraType shuffle_multiply(
            ConstRawUnspecifiedAlgebraType left,
            ConstRawUnspecifiedAlgebraType right
            ) const;

    virtual UnspecifiedAlgebraType half_shuffle_multiply(
            ConstRawUnspecifiedAlgebraType left,
            ConstRawUnspecifiedAlgebraType right
            ) const;

    virtual UnspecifiedAlgebraType adjoint_to_left_multiply_by(
            ConstRawUnspecifiedAlgebraType multiplier,
            ConstRawUnspecifiedAlgebraType argument
            ) const;



};

RPY_EXPORT base_context_pointer get_base_context(deg_t width, deg_t depth);

RPY_EXPORT context_pointer get_context(
        deg_t width, deg_t depth, const scalars::ScalarType* ctype,
        const std::vector<std::pair<string, string>>& preferences = {}
);

inline void check_contexts_compatible(const Context& ctx1, const Context& ctx2)
{
    if (&ctx1 == &ctx2) {
        // Early exit if both reference the same object.
        return;
    }

    if (ctx1.width() != ctx2.width()) {
        RPY_THROW(std::invalid_argument, "contexts have incompatible width");
    }

    const auto* ctype1 = ctx1.ctype();
    const auto* ctype2 = ctx2.ctype();
    if (ctype1 == ctype2) {
        // Both are OK if the ctypes are identical
        return;
    }

    // TODO: Check that ctypes are actually same type but potentially different
    // locations
    // TODO: Alternatively, check that ctype1 is convertible to ctype2
}

class RPY_EXPORT ContextMaker
{
public:
    using preference_list = std::vector<std::pair<string, string>>;

    virtual ~ContextMaker() = default;
    virtual bool
    can_get(deg_t width, deg_t depth, const scalars::ScalarType* ctype,
            const preference_list& preferences) const;
    virtual context_pointer get_context(
            deg_t width, deg_t depth, const scalars::ScalarType* ctype,
            const preference_list& preferences
    ) const = 0;
    virtual optional<base_context_pointer>
    get_base_context(deg_t width, deg_t depth) const = 0;
};

RPY_EXPORT const ContextMaker*
register_context_maker(std::unique_ptr<ContextMaker> maker);

template <typename Maker>
class RegisterMakerHelper
{
    const ContextMaker* maker = nullptr;

public:
    template <typename... Args>
    explicit RegisterMakerHelper(Args&&... args)
    {
        maker = register_context_maker(std::unique_ptr<ContextMaker>(
                new Maker(std::forward<Args>(args)...)
        ));
    }
};

#define RPY_ALGEBRA_DECLARE_CTX_MAKER(MAKER)                                   \
    static RegisterMakerHelper<MAKER> rpy_static_algebra_maker_decl_##MAKER    \
            = RegisterMakerHelper<MAKER>()

inline bool check_contexts_algebra_compatible(
        const Context& base, const Context& other
) noexcept
{
    if (base.width() != other.width()) { return false; }

    if (other.depth() < base.depth()) { return false; }

    return true;
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_CONTEXT_H_

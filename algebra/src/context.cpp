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

//
// Created by user on 05/03/23.
//

#include <roughpy/algebra/context.h>

#include "hall_set_size.h"

#include <mutex>
#include <unordered_map>

#include <boost/functional/hash.hpp>

#include "lite_context.h"

using namespace rpy;
using namespace rpy::algebra;

BasicContextSpec rpy::algebra::get_context_spec(const context_pointer& ctx)
{
    if (!ctx) { return {"", "", 0, 0}; }
    return {ctx->ctype()->id(), ctx->backend(), ctx->width(), ctx->depth()};
}

context_pointer rpy::algebra::from_context_spec(const BasicContextSpec& spec)
{
    RPY_CHECK(spec.stype_id != "");

    return get_context(
            spec.width, spec.depth, scalars::get_type(spec.stype_id),
            {
                    {"backend", spec.backend}
    }
    );
}

std::vector<byte> rpy::algebra::alg_to_raw_bytes(
        context_pointer ctx, AlgebraType atype, RawUnspecifiedAlgebraType alg
)
{
    if (ctx) { return ctx->to_raw_bytes(atype, alg); }
    return std::vector<byte>();
}

UnspecifiedAlgebraType rpy::algebra::alg_from_raw_bytes(
        context_pointer ctx, AlgebraType atype, Slice<byte> raw_data
)
{
    if (ctx) { return ctx->from_raw_bytes(atype, raw_data); }
    return nullptr;
}

ContextBase::ContextBase(
        deg_t width, deg_t depth, const dimn_t* lie_sizes,
        const dimn_t* tensor_sizes
)
    : m_width(width), m_depth(depth), p_lie_sizes(lie_sizes),
      p_tensor_sizes(tensor_sizes)
{
    if (!p_tensor_sizes) {
        auto* tsizes = new dimn_t[1 + m_depth];
        // Immediately give ownership to the MaybeOwned type so we don't leak
        // memory.
        p_tensor_sizes = tsizes;
        tsizes[0] = 1;
        tsizes[1] = 1 + m_width;
        for (deg_t i = 1; i <= m_depth; ++i) {
            tsizes[i] = 1 + tsizes[i - 1] * m_width;
        }
    }

    if (!p_lie_sizes) {
        HallSetSizeHelper helper(m_width, m_depth);
        auto* lsizes = new dimn_t[1 + m_depth];
        // Immediately give ownership to the MaybeOwned type so we don't leak
        // memory.
        p_lie_sizes = lsizes;
        lsizes[0] = 0;
        lsizes[1] = m_width;
        for (int i = 2; i <= m_depth; ++i) {
            lsizes[i] = helper(i) * lsizes[i - 1];
        }
    }
}

ContextBase::~ContextBase() = default;

dimn_t ContextBase::lie_size(deg_t deg) const noexcept
{
    if (deg < 0 || deg > m_depth) { deg = m_depth; }
    return p_lie_sizes[deg];
}
dimn_t ContextBase::tensor_size(deg_t deg) const noexcept
{
    if (deg < 0 || deg > m_depth) { deg = m_depth; }
    return p_tensor_sizes[deg];
}
bool Context::check_compatible(const Context& other_ctx) const noexcept
{
    return width() == other_ctx.width();
}
FreeTensor Context::zero_free_tensor(VectorType vtype) const
{
    return construct_free_tensor({scalars::KeyScalarArray(), vtype});
}
ShuffleTensor Context::zero_shuffle_tensor(VectorType vtype) const
{
    return construct_shuffle_tensor({scalars::KeyScalarArray(), vtype});
}
Lie Context::zero_lie(VectorType vtype) const
{
    return construct_lie({scalars::KeyScalarArray(), vtype});
}
FreeTensor Context::to_signature(const Lie& log_signature) const
{
    return lie_to_tensor(log_signature).exp();
}

void Context::lie_to_tensor_fallback(FreeTensor& result, const Lie& arg) const
{
    // TODO: Implement
}
void Context::tensor_to_lie_fallback(Lie& result, const FreeTensor& arg) const
{
    // TODO: Implement
}
void Context::cbh_fallback(FreeTensor& collector, const std::vector<Lie>& lies)
        const
{
    for (const auto& alie : lies) {
        if (alie.dimension() != 0) {
            collector.fmexp(this->lie_to_tensor(alie));
        }
    }
}

Lie Context::cbh(const std::vector<Lie>& lies, VectorType vtype) const
{
    if (lies.size() == 1) { return convert(lies[0], vtype); }

    FreeTensor collector = zero_free_tensor(vtype);
    collector[0] = scalars::Scalar(1);

    if (!lies.empty()) { cbh_fallback(collector, lies); }

    return tensor_to_lie(collector.log());
}
Lie Context::cbh(const Lie& left, const Lie& right, VectorType vtype) const
{
    FreeTensor tmp = lie_to_tensor(left).exp();
    tmp.fmexp(lie_to_tensor(right));

    return tensor_to_lie(tmp.log());
}

static std::recursive_mutex s_context_lock;

static std::vector<std::unique_ptr<ContextMaker>>& get_context_maker_list()
{
    static std::vector<std::unique_ptr<ContextMaker>> list;
    return list;
}

namespace {

class ConcreteContextBase : public ContextBase
{
public:
    ConcreteContextBase(deg_t width, deg_t depth)
        : ContextBase(width, depth, nullptr, nullptr)
    {}
};

}// namespace

base_context_pointer rpy::algebra::get_base_context(deg_t width, deg_t depth)
{
    std::lock_guard<std::recursive_mutex> access(s_context_lock);
    auto& maker_list = get_context_maker_list();

    for (const auto& maker : maker_list) {
        auto found = maker->get_base_context(width, depth);
        if (found.has_value()) { return *found; }
    }

    // No context makers have a base context with this configuration, let's
    // make a new one
    static std::vector<base_context_pointer> s_base_context_cache;

    for (const auto& bcp : s_base_context_cache) {
        if (bcp->width() == width && bcp->depth() == depth) { return bcp; }
    }

    s_base_context_cache.emplace_back(new ConcreteContextBase(width, depth));
    return s_base_context_cache.back();
}
context_pointer rpy::algebra::get_context(
        deg_t width, deg_t depth, const scalars::ScalarType* ctype,
        const std::vector<std::pair<string, string>>& preferences
)
{
    std::lock_guard<std::recursive_mutex> access(s_context_lock);
    auto& maker_list = get_context_maker_list();

    if (maker_list.empty()) { maker_list.emplace_back(new LiteContextMaker); }

    std::vector<const ContextMaker*> found;
    found.reserve(maker_list.size());
    for (const auto& maker : maker_list) {
        if (maker->can_get(width, depth, ctype, preferences)) {
            found.push_back(maker.get());
        }
    }

    if (found.empty()) {
        RPY_THROW(std::invalid_argument,"cannot find a context maker for the "
                                    "width, depth, dtype, and preferences set");
    }

    if (found.size() > 1) {
        RPY_THROW(std::invalid_argument,
                "found multiple context maker candidates for specified width, "
                "depth, dtype, and preferences set"
        );
    }

    return found[0]->get_context(width, depth, ctype, preferences);
}

const ContextMaker*
rpy::algebra::register_context_maker(std::unique_ptr<ContextMaker> maker)
{
    std::lock_guard<std::recursive_mutex> access(s_context_lock);
    auto& maker_list = get_context_maker_list();
    maker_list.push_back(std::move(maker));
    return maker_list.back().get();
}

bool ContextMaker::can_get(
        deg_t width, deg_t depth, const scalars::ScalarType* ctype,
        const preference_list& preferences
) const
{
    return false;
}

std::vector<byte>
Context::to_raw_bytes(AlgebraType atype, RawUnspecifiedAlgebraType alg) const
{
    RPY_THROW(std::runtime_error, "cannot generate raw byte representation");
}
UnspecifiedAlgebraType
Context::from_raw_bytes(AlgebraType atype, Slice<byte> raw_bytes) const
{
    RPY_THROW(std::runtime_error, "cannot load from raw bytes");
}

void rpy::algebra::intrusive_ptr_release(const rpy::algebra::Context* ptr)
{
    intrusive_ptr_release(static_cast<const boost::intrusive_ref_counter<
                                  ContextBase, boost::thread_safe_counter>*>(ptr
    ));
}
void rpy::algebra::intrusive_ptr_release(const rpy::algebra::ContextBase* ptr)
{
    intrusive_ptr_release(static_cast<const boost::intrusive_ref_counter<
                                  ContextBase, boost::thread_safe_counter>*>(ptr
    ));
}

void rpy::algebra::intrusive_ptr_add_ref(const rpy::algebra::Context* ptr)
{
    intrusive_ptr_add_ref(static_cast<const boost::intrusive_ref_counter<
                                  ContextBase, boost::thread_safe_counter>*>(ptr
    ));
}
void rpy::algebra::intrusive_ptr_add_ref(const rpy::algebra::ContextBase* ptr)
{
    intrusive_ptr_add_ref(static_cast<const boost::intrusive_ref_counter<
                                  ContextBase, boost::thread_safe_counter>*>(ptr
    ));
}

UnspecifiedAlgebraType Context::free_multiply(
        const ConstRawUnspecifiedAlgebraType left,
        const ConstRawUnspecifiedAlgebraType right
) const
{
    RPY_THROW(std::runtime_error,"free tensor multiply is not implemented for "
                                         "arbitrary types with this backend");
}
UnspecifiedAlgebraType Context::shuffle_multiply(
        ConstRawUnspecifiedAlgebraType left,
        ConstRawUnspecifiedAlgebraType right
) const
{
    RPY_THROW(std::runtime_error,"shuffle multiply is not implemented for "
                                         "arbitrary types with this backend");
}
UnspecifiedAlgebraType Context::half_shuffle_multiply(
        ConstRawUnspecifiedAlgebraType left,
        ConstRawUnspecifiedAlgebraType right
) const
{
    RPY_THROW(std::runtime_error,"half shuffle multiply is not implemented for "
                             "arbitrary types with this backend");
}
UnspecifiedAlgebraType Context::adjoint_to_left_multiply_by(
        ConstRawUnspecifiedAlgebraType multiplier,
        ConstRawUnspecifiedAlgebraType argument
) const
{
    RPY_THROW(std::runtime_error,"adjoint of left multiply is not implemented for "
                             "arbitrary types with this backend");
}

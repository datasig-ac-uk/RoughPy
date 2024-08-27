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

#include <mutex>
#include <unordered_map>

using namespace rpy;
using namespace rpy::algebra;

BasicContextSpec rpy::algebra::get_context_spec(const context_pointer& ctx)
{
    if (!ctx) { return {"", "", 0, 0}; }
    return {string(ctx->ctype()->id()),
            ctx->backend(),
            ctx->width(),
            ctx->depth()};
}

context_pointer rpy::algebra::from_context_spec(const BasicContextSpec& spec)
{
    RPY_CHECK(spec.stype_id != "");

    auto tp_o = scalars::get_type(spec.stype_id);

    RPY_CHECK(tp_o);
    return get_context(
            spec.width,
            spec.depth,
            tp_o,
            {
                    {"backend", spec.backend}
    }
    );
}

static std::recursive_mutex s_context_lock;

static containers::Vec<std::unique_ptr<ContextMaker>>& get_context_maker_list()
{
    static containers::Vec<std::unique_ptr<ContextMaker>> list;
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
    static containers::Vec<base_context_pointer> s_base_context_cache;

    for (const auto& bcp : s_base_context_cache) {
        if (bcp->width() == width && bcp->depth() == depth) { return bcp; }
    }

    s_base_context_cache.emplace_back(new ConcreteContextBase(width, depth));
    return s_base_context_cache.back();
}
context_pointer rpy::algebra::get_context(
        deg_t width,
        deg_t depth,
        scalars::TypePtr ctype,
        const containers::Vec<std::pair<string, string>>& preferences
)
{
    std::lock_guard<std::recursive_mutex> access(s_context_lock);
    auto& maker_list = get_context_maker_list();

    containers::Vec<const ContextMaker*> found;
    found.reserve(maker_list.size());
    for (const auto& maker : maker_list) {
        if (maker->can_get(width, depth, ctype, preferences)) {
            found.push_back(maker.get());
        }
    }

    if (found.empty()) {
        RPY_THROW(
                std::invalid_argument,
                "cannot find a context maker for the "
                "width, depth, dtype, and preferences set"
        );
    }

    if (found.size() > 1) {
        RPY_THROW(
                std::invalid_argument,
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
        deg_t width,
        deg_t depth,
        scalars::TypePtr ctype,
        const preference_list& preferences
) const
{
    return false;
}

void rpy::algebra::intrusive_ptr_add_ref(const ContextBase* ptr) noexcept
{
    using counter_t = boost::
            intrusive_ref_counter<ContextBase, boost::thread_safe_counter>;
    intrusive_ptr_add_ref(static_cast<const counter_t*>(ptr));
}
void rpy::algebra::intrusive_ptr_release(const ContextBase* ptr) noexcept
{
    using counter_t = boost::
            intrusive_ref_counter<ContextBase, boost::thread_safe_counter>;
    intrusive_ptr_release(static_cast<const counter_t*>(ptr));
}
void rpy::algebra::intrusive_ptr_add_ref(const Context* ptr) noexcept
{
    using counter_t = boost::
            intrusive_ref_counter<ContextBase, boost::thread_safe_counter>;
    intrusive_ptr_add_ref(static_cast<const counter_t*>(ptr));
}
void rpy::algebra::intrusive_ptr_release(const Context* ptr) noexcept
{
    using counter_t = boost::
            intrusive_ref_counter<ContextBase, boost::thread_safe_counter>;
    intrusive_ptr_release(static_cast<const counter_t*>(ptr));
}

Lie Context::zero_lie(VectorType) const { return Lie(); }
FreeTensor Context::zero_free_tensor(VectorType) const { return FreeTensor(); }
ShuffleTensor Context::zero_shuffle_tensor(VectorType) const
{
    return ShuffleTensor();
}

Lie Context::cbh(const Lie& lhs, const Lie& rhs, VectorType) const
{
    return Lie();
}
Lie Context::cbh(Slice<Lie> lies, VectorType) const { return Lie(); }

FreeTensor Context::to_signature(const Lie& log_signature) const
{
    return FreeTensor();
}
ContextBase::ContextBase(
        deg_t width,
        deg_t depth,
        const dimn_t* lie_sizes,
        const dimn_t* tensor_sizes
)
    : m_width(width),
      m_depth(depth)
{}
ContextBase::~ContextBase() {}
dimn_t ContextBase::lie_size(deg_t deg) const noexcept { return 0; }
dimn_t ContextBase::tensor_size(deg_t deg) const noexcept { return 0; }

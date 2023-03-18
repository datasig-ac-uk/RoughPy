//
// Created by user on 06/03/23.
//

#include "lite_context.h"
#include <roughpy/config/macros.h>

#include "context.h"

#include <array>
#include <unordered_map>


using namespace rpy;
using namespace rpy::algebra;



RPY_ALGEBRA_DECLARE_CTX_MAKER(LiteContextMaker);


static std::unordered_map<std::tuple<deg_t, deg_t, const scalars::ScalarType*>, context_pointer,
    boost::hash<std::tuple<deg_t, deg_t, const scalars::ScalarType*>>> s_lite_context_cache;

static std::array<const scalars::ScalarType *, 3> s_lite_context_allowed_ctypes = {
    scalars::ScalarType::of<double>(),
    scalars::ScalarType::of<float>(),
    scalars::ScalarType::of<typename lal::rational_field::scalar_type>()};


static optional<std::ptrdiff_t> index_of_ctype(const scalars::ScalarType* ctype) noexcept {
    auto begin = s_lite_context_allowed_ctypes.begin();
    auto end = s_lite_context_allowed_ctypes.end();

    auto found = std::find(begin, end, ctype);
    if (found == end) {
        return {};
    }
    return static_cast<std::ptrdiff_t>(found - begin);
}

bool LiteContextMaker::can_get(deg_t width, deg_t depth, const scalars::ScalarType *ctype, const preference_list &preferences) const {

    if (!index_of_ctype(ctype).has_value()) {
        return false;
    }

    {
        auto begin = preferences.begin();
        auto end = preferences.end();

        if (!preferences.empty()) {
            auto it = std::find_if(begin, end, [](auto p) { return p.first == "backend"; });
            if (it != end && it->second != "libalgebra_lite") {
                return false;
            }

        }
    }

    return true;
}

context_pointer LiteContextMaker::create_context(deg_t width, deg_t depth, const scalars::ScalarType *ctype, const ContextMaker::preference_list &preferences) const {
    auto idx = index_of_ctype(ctype);
    assert(idx.has_value());

    switch (idx.value()) {
        case 0: return new LiteContext<lal::double_field>(width, depth);
        case 1: return new LiteContext<lal::float_field>(width, depth);
        case 2: return new LiteContext<lal::rational_field>(width, depth);
    }

    RPY_UNREACHABLE();
}

context_pointer LiteContextMaker::get_context(deg_t width, deg_t depth, const scalars::ScalarType *ctype, const preference_list &preferences) const {
    auto& found = s_lite_context_cache[std::make_tuple(width, depth, ctype)];

    if (!found) {
        found = create_context(width, depth, ctype, preferences);
    }

    return found;
}
optional<base_context_pointer> LiteContextMaker::get_base_context(deg_t width, deg_t depth) const {
    auto begin = s_lite_context_cache.begin();
    auto end = s_lite_context_cache.end();
    auto found = std::find_if(begin, end, [width, depth](auto p) {
        return std::get<0>(p.first) == width && std::get<1>(p.first) == depth;
    });

    if (found != end) {
        return static_cast<base_context_pointer>(found->second);
    }


    return {};
}

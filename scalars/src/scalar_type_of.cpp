//
// Created by sam on 11/15/23.
//

#include "scalars_fwd.h"

#include "builtin_scalar_types.h"
#include "scalar_type.h"
#include "scalar_type_of.h"

#include "devices/core.h"
#include "devices/host_device.h"

#include <roughpy/core/container/unordered_map.h>

#include <charconv>

using namespace rpy;
using namespace rpy::scalars;

//
// PackedScalarType rpy::scalars::scalar_type_of(devices::TypeInfo info)
// {
// #define X(TP) return scalar_type_of<TP>()
//     DO_FOR_EACH_X(info)
// #undef X
//
//     RPY_THROW(std::runtime_error, "unsupported scalar type");
// }
//

namespace {
inline optional<devices::TypeInfo>
parse_byted_type(string_view id_sub, devices::TypeCode code) noexcept
{
    dimn_t bits = 0;

    auto result = std::from_chars(&*id_sub.begin(), &*id_sub.end(), bits);
    if (result.ec != std::errc{}) { return {}; }

    const auto bytes = static_cast<uint8_t>(bits / CHAR_BIT);

    return devices::TypeInfo{code, bytes, next_power_2(bytes), 1};
}

inline optional<devices::TypeInfo> parse_id_to_type_info(string_view id
) noexcept
{
    using devices::TypeCode;
    RPY_DBG_ASSERT(!id.empty());

    switch (id.front()) {
        case 'i': return parse_byted_type(id.substr(1), TypeCode::Int);
        case 'u': return parse_byted_type(id.substr(1), TypeCode::UInt);
        case 'f': return parse_byted_type(id.substr(1), TypeCode::Float);
        case 'c': return parse_byted_type(id.substr(1), TypeCode::Complex);
        case 'b':
            if (id.size() > 1 && id[2] == 'f') {
                return parse_byted_type(id.substr(2), TypeCode::BFloat);
            }
        default: break;
    }

    return {};
}
}// namespace

static std::recursive_mutex s_type_cache_lock;
static containers::HashMap<string_view, const ScalarType*> s_scalar_type_cache{
        {         "f16",             HalfType::get()},
        {         "f32",            FloatType::get()},
        {         "f64",           DoubleType::get()},
        {    "Rational", APRationalScalarType::get()},
        {        "bf16",   BFloat16ScalarType::get()},
        {"RationalPoly",  APRatPolyScalarType::get()}
};

PackedScalarType scalars::scalar_type_of(string_view id, devices::Device device)
{
    {
        std::lock_guard<std::recursive_mutex> access(s_type_cache_lock);
        auto it = s_scalar_type_cache.find(id);
        if (it != s_scalar_type_cache.end()) {
            if (device && !device->type() == devices::DeviceType::CPU) {
                return it->second->with_device(device);
            }
            return it->second;
        }
    }

    if (auto info = parse_id_to_type_info(id)) { return *info; }

    RPY_THROW(
            std::runtime_error,
            "no scalar type with id \"" + string(id) + '\"'
    );
}

optional<const ScalarType*> scalars::get_type(string_view id)
{
    std::lock_guard access(s_type_cache_lock);
    const auto it = s_scalar_type_cache.find(id);
    if (it != s_scalar_type_cache.end()) {
        RPY_DBG_ASSERT(it->second != nullptr);
        return it->second;
    }
    return {};
}

void scalars::register_scalar_type(const ScalarType* tp)
{
    RPY_CHECK(tp != nullptr);
    std::lock_guard<std::recursive_mutex> access(s_type_cache_lock);

    auto& elt = s_scalar_type_cache[tp->id()];
    if (elt == nullptr) { elt = tp; }
}
void scalars::unregister_scalar_type(string_view id)
{
    std::lock_guard<std::recursive_mutex> access(s_type_cache_lock);

    auto it = s_scalar_type_cache.find(id);
    RPY_CHECK(
            it != s_scalar_type_cache.end(),
            "no type with id \"" + string(id) + '\"'
    );

    s_scalar_type_cache.erase(it);
}

//
// Created by sam on 3/30/24.
//

#include "devices/type.h"

#include <roughpy/core/container/unordered_map.h>
#include <roughpy/platform/alloc.h>

#include "devices/device_handle.h"

// ReSharper disable CppUnusedIncludeDirective
#include "fundamental_types/int16_type.h"
#include "fundamental_types/int32_type.h"
#include "fundamental_types/int64_type.h"
#include "fundamental_types/int8_type.h"

#include "fundamental_types/uint16_type.h"
#include "fundamental_types/uint32_type.h"
#include "fundamental_types/uint64_type.h"
#include "fundamental_types/uint8_type.h"

#include "fundamental_types/double_type.h"
#include "fundamental_types/float_type.h"
// ReSharper restore CppUnusedIncludeDirective

#include "roughpy/core/strings.h"

#include <typeindex>

using namespace rpy;
using namespace rpy::devices;

Type::Type(string_view id, string_view name, TypeInfo info, TypeTraits traits)
    : m_id(std::move(id)),
      m_name(std::move(name)),
      m_info(info),
      m_traits(traits)
{
    register_type(this);
}

Type::~Type() = default;

Buffer Type::allocate(Device device, dimn_t count) const
{
    return device->alloc(this, count);
}

void* Type::allocate_single() const
{
    return aligned_alloc(m_info.alignment, m_info.bytes);
}

void Type::free_single(void* ptr) const { aligned_free(ptr); }

bool Type::supports_device(const Device& device) const noexcept { return true; }

const Type* devices::get_type(TypeInfo info)
{
    switch (info.code) {
        case TypeCode::Int:
            switch (info.bytes) {
                case 1: return FundamentalType<int8_t>::get();
                case 2: return FundamentalType<int16_t>::get();
                case 4: return FundamentalType<int32_t>::get();
                case 8: return FundamentalType<int64_t>::get();
                default: break;
            }
        case TypeCode::UInt:
            switch (info.bytes) {
                case 1: return FundamentalType<uint8_t>::get();
                case 2: return FundamentalType<uint16_t>::get();
                case 4: return FundamentalType<uint32_t>::get();
                case 8: return FundamentalType<uint64_t>::get();
                default: break;
            }
        case TypeCode::Float:
            switch (info.bytes) {
                case 4: return FloatType::get();
                case 8: return DoubleType::get();
                default: break;
            }
        default: break;
    }

    RPY_THROW(
            std::runtime_error,
            "Only fundamental types can be read from TypeInfo"
    );
}

// #ifndef RPY_NO_RTTI
//
// namespace {
//
// struct Protected {
//     containers::FlatHashMap<std::type_index, const Type*>& map;
//     std::lock_guard<std::recursive_mutex> access;
// };
//
// Protected get_cache() noexcept
// {
//     static std::recursive_mutex m_lock;
//     static containers::FlatHashMap<std::type_index, const Type*> s_map;
//     // {
//     //     {typeid(float), FundamentalType<float>::get()},
//     //     {typeid(double), FundamentalType<double>::get()},
//     //     {typeid(int8_t), FundamentalType<int8_t>::get()},
//     //     {typeid(int16_t), FundamentalType<int16_t>::get()},
//     //     {typeid(int32_t), FundamentalType<int32_t>::get()},
//     //     {typeid(int64_t), FundamentalType<int64_t>::get()},
//     //     {typeid(uint8_t), FundamentalType<uint8_t>::get()},
//     //     {typeid(uint16_t), FundamentalType<uint16_t>::get()},
//     //     {typeid(uint32_t), FundamentalType<uint32_t>::get()},
//     //     {typeid(uint64_t), FundamentalType<uint64_t>::get()},
//     // };
//     //
//     return {s_map, std::lock_guard(m_lock)};
// }
//
// }// namespace

// void devices::register_type(const std::type_info& info, const Type* type)
// {
//     auto cache = get_cache();
//     auto& entry = cache.map[std::type_index(info)];
//     if (entry != nullptr) {
//         RPY_THROW(
//                 std::runtime_error,
//                 "type " + string(info.name()) + " is already registered"
//         );
//     }
//
//     entry = type;
// }
//
// const Type* devices::get_type(const std::type_info& info)
// {
//     const auto cache = get_cache();
//
//     const auto it = cache.map.find(std::type_index(info));
//     if (it == cache.map.end()) {
//         RPY_THROW(
//                 std::runtime_error,
//                 "no type matching type id " + string(info.name())
//         );
//     }
//
//     return it->second;
// }
//
// #endif

bool Type::convertible_to(const Type* dest_type) const noexcept
{
    return dest_type->convertible_from(this);
}
bool Type::convertible_from(const Type* src_type) const noexcept
{
    if (src_type == this) { return true; }
    if (traits::is_arithmetic(src_type)) { return true; }

    return false;
}
TypeComparison Type::compare_with(const Type* other) const noexcept
{
    if (other == this) { return TypeComparison::AreSame; }

    if (traits::is_arithmetic(other)) { return TypeComparison::Convertible; }

    return TypeComparison::NotConvertible;
}

namespace {

template <typename Guarded, typename Mutex = std::mutex>
class MutexLocked
{
    Guarded& r_guarded;
    std::scoped_lock<Mutex> m_guard;

public:
    using mutex_type = Mutex;

    MutexLocked(Guarded& guarded, Mutex& mutex)
        : r_guarded(guarded),
          m_guard(mutex)
    {}

    Guarded& value() noexcept { return r_guarded; }
};

using TypeCache = containers::HashMap<string_view, const Type*>;
using GuardedTypeCache = MutexLocked<TypeCache>;

GuardedTypeCache get_type_cache() noexcept
{
    static typename GuardedTypeCache::mutex_type lock;
    static TypeCache s_cache;
    return {s_cache, lock};
}

}// namespace

void devices::register_type(const Type* tp)
{
    auto locked_cache = get_type_cache();
    auto& cache = locked_cache.value();

    auto& entry = cache[tp->id()];
    if (entry == nullptr) {
        entry = tp;
    } else {
        RPY_THROW(
                std::runtime_error,
                string_cat("type with id ", tp->id(), " already exists")
        );
    }
}

const Type* devices::get_type(string_view type_id)
{
    auto locked_cache = get_type_cache();
    auto& cache = locked_cache.value();

    const auto found = cache.find(type_id);
    if (found != cache.end()) { return found->second; }

    RPY_THROW(
            std::runtime_error,
            string_cat("type id ", type_id, " not found")
    );
}

namespace {

struct InitializeAllTheFundamentals
{
    InitializeAllTheFundamentals()
    {
        volatile const Type* ptr;
        ptr = FundamentalType<int8_t>::get();
        ptr = FundamentalType<int16_t>::get();
        ptr = FundamentalType<int32_t>::get();
        ptr = FundamentalType<int64_t>::get();

        ptr = FundamentalType<uint8_t>::get();
        ptr = FundamentalType<uint16_t>::get();
        ptr = FundamentalType<uint32_t>::get();
        ptr = FundamentalType<uint64_t>::get();

        ptr = FloatType::get();
        ptr = DoubleType::get();

        (void) ptr;
    }
};

}

static InitializeAllTheFundamentals s_fundamentals {};
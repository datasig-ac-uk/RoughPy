//
// Created by user on 26/02/23.
//

#include "scalar_type.h"

#include "scalar_pointer.h"
#include "scalar.h"

#include <mutex>
#include <ostream>
#include <unordered_map>


#include "float_type.h"
#include "double_type.h"
#include "RationalType.h"

using namespace rpy;
using namespace scalars;

const ScalarType *ScalarType::rational_type() const noexcept {
    return this;
}

const ScalarType *ScalarType::for_id(const std::string &id) {
    return ScalarType::of<double>();
}
const ScalarType *ScalarType::from_type_details(const BasicScalarInfo &details, const ScalarDeviceInfo& device) {
    return ScalarType::of<double>();
}
Scalar ScalarType::from(long long int numerator, long long int denominator) const {
    if (denominator == 0){
        throw std::invalid_argument("division by zero");
    }
    auto result = allocate(1);
    assign(result, numerator, denominator);
    return {result, Scalar::OwnedPointer};
}
void ScalarType::convert_fill(ScalarPointer out, ScalarPointer in, dimn_t count, const std::string &id) const {
    if (!id.empty()) {
        try {
            const auto& conversion = get_conversion(id, this->id());
            conversion(out, in, count);
            return;
        } catch (std::runtime_error&) {
        }
    }

    convert_copy(out, in, 1);

    auto isize = itemsize();
    auto* out_p = out.raw_cast<char*>();
    for (dimn_t i=1; i<count; ++i) {
        out_p += isize;
        convert_copy({this, out_p}, out, 1);
    }
}
Scalar ScalarType::one() const {
    return from(1, 1);
}
Scalar ScalarType::mone() const {
    return from(-1, 1);
}
Scalar ScalarType::zero() const {
    return from(0, 1);
}
Scalar ScalarType::copy(ScalarPointer source) const {
    auto result = allocate(1);
    convert_copy(result, source, 1);

    return {result, Scalar::OwnedPointer};
}
Scalar ScalarType::add(ScalarPointer lhs, ScalarPointer rhs) const {
    auto result = copy(lhs);
    add_inplace(result.to_mut_pointer(), rhs);
    return result;
}
Scalar ScalarType::sub(ScalarPointer lhs, ScalarPointer rhs) const {
    auto result = copy(lhs);
    sub_inplace(result.to_mut_pointer(), rhs);
    return result;
}
Scalar ScalarType::mul(ScalarPointer lhs, ScalarPointer rhs) const {
    auto result = copy(lhs);
    mul_inplace(result.to_mut_pointer(), rhs);
    return result;
}
Scalar ScalarType::div(ScalarPointer lhs, ScalarPointer rhs) const {
    auto result = copy(lhs);
    div_inplace(result.to_mut_pointer(), rhs);
    return result;
}

bool ScalarType::is_zero(ScalarPointer arg) const {
    return arg.is_null() || are_equal(arg, zero().to_pointer());
}
void ScalarType::print(ScalarPointer arg, std::ostream &os) const {
    os << to_scalar_t(arg);
}

template<>
const ScalarType* rpy::scalars::dtl::scalar_type_holder<float>::get_type() noexcept {
    static const FloatType ftype;
    return &ftype;
}

template <>
const ScalarType* rpy::scalars::dtl::scalar_type_holder<double>::get_type() noexcept {
    static const DoubleType dtype;
    return &dtype;
}

template <>
const ScalarType* rpy::scalars::dtl::scalar_type_holder<rational_scalar_type>::get_type() noexcept {
    static const RationalType rtype;
    return &rtype;
}

/*
 * All types should support conversion from the following,
 * but these are not represented as scalar types by themselves.
 * The last 3 are commented out, because they are to be implemented
 * separately.
 */
static const std::string reserved[] = {
    "i32",  // int
    "u32",  // unsigned int
    "i64",  // long long
    "u64",  // unsigned long long
    //    "l",                // long
    //    "L",                // unsigned long
    "isize",// ssize_t
    "usize",// size_t
    "i16",  // short
    "u16",  // unsigned short
    "i8",   // char
    "u8",   // unsigned char
    //    "c",                // char
    //    "e",                // float16
    //    "g",                // float128
    //    "O"                 // Object
};
static std::mutex s_scalar_type_cache_lock;
static std::unordered_map<std::string, const ScalarType *> s_scalar_type_cache {
    {std::string("f32"), ScalarType::of<float>()},
    {std::string("f64"), ScalarType::of<float>()}
//    {std::string("rational"), ScalarType::of<float>()},
};

void rpy::scalars::register_type(const ScalarType *type) {
    std::lock_guard<std::mutex> access(s_scalar_type_cache_lock);

    const auto& identifier = type->id();
    for (const auto& i : reserved) {
        if (identifier == i) {
            throw std::runtime_error("cannot register identifier " + identifier + ", it is reserved");
        }
    }

    auto& entry = s_scalar_type_cache[identifier];
    if (entry != nullptr) {
        throw std::runtime_error("type with id " + identifier + " is already registered");
    }

    entry = type;
}
const ScalarType *rpy::scalars::get_type(const std::string &id) {
    return nullptr;
}

std::vector<const ScalarType *> rpy::scalars::list_types() {
    return std::vector<const ScalarType *>();
}

#define MAKE_CONVERSION_FUNCTION(SRC, DST, SRC_T, DST_T)                             \
    static void SRC##_to_##DST(ScalarPointer dst, ScalarPointer src, dimn_t count) { \
        const auto *src_p = src.raw_cast<const SRC_T>();                             \
        auto *dst_p = dst.raw_cast<DST_T>();                                     \
                                                                                     \
        for (dimn_t i = 0; i < count; ++i) {                                         \
            ::new (dst_p++) DST_T(src_p[i]);                                         \
        }                                                                            \
    }

MAKE_CONVERSION_FUNCTION(f32, f64, float, double)
MAKE_CONVERSION_FUNCTION(f64, f32, double, float)
MAKE_CONVERSION_FUNCTION(i32, f32, int, float)
MAKE_CONVERSION_FUNCTION(i32, f64, int, double)
MAKE_CONVERSION_FUNCTION(i64, f32, long long, float)
MAKE_CONVERSION_FUNCTION(i64, f64, long long, double)
MAKE_CONVERSION_FUNCTION(i16, f32, short, float)
MAKE_CONVERSION_FUNCTION(i16, f64, short, double)
MAKE_CONVERSION_FUNCTION(i8, f32, char, float)
MAKE_CONVERSION_FUNCTION(i8, f64, char, double)
MAKE_CONVERSION_FUNCTION(isize, f32, idimn_t, float)
MAKE_CONVERSION_FUNCTION(isize, f64, idimn_t, double)

#undef MAKE_CONVERSION_FUNCTION


using pair_type = std::pair<std::string, conversion_function>;

#define ADD_DEF_CONV(SRC, DST) \
    pair_type { std::string(#SRC "->" #DST), conversion_function(&SRC##_to_##DST) }

static std::mutex s_conversion_lock;
static std::unordered_map<std::string, conversion_function> s_conversion_cache{
    ADD_DEF_CONV(f32, f64),
    ADD_DEF_CONV(f64, f32),
    ADD_DEF_CONV(i32, f32),
    ADD_DEF_CONV(i32, f64),
    ADD_DEF_CONV(i64, f32),
    ADD_DEF_CONV(i64, f64),
    ADD_DEF_CONV(i16, f32),
    ADD_DEF_CONV(i16, f64),
    ADD_DEF_CONV(i8, f32),
    ADD_DEF_CONV(i8, f64),
    ADD_DEF_CONV(isize, f32),
    ADD_DEF_CONV(isize, f64)};

#undef ADD_DEF_CONV

static inline std::string type_ids_to_key(const std::string &src_type, const std::string &dst_type) {
    return src_type + "->" + dst_type;
}

const conversion_function &rpy::scalars::get_conversion(const std::string &src_id, const std::string &dst_id) {
    std::lock_guard<std::mutex> access(s_conversion_lock);

    auto found = s_conversion_cache.find(type_ids_to_key(src_id, dst_id));
    if (found != s_conversion_cache.end()) {
        return found->second;
    }

    throw std::runtime_error("no conversion function from " + src_id + " to " + dst_id);
}
void rpy::scalars::register_conversion(const std::string &src_id, const std::string &dst_id, conversion_function converter) {
    std::lock_guard<std::mutex> access(s_conversion_lock);

    auto &found = s_conversion_cache[type_ids_to_key(src_id, dst_id)];
    if (found != nullptr) {
        throw std::runtime_error("conversion from " + src_id + " to " + dst_id + " already registered");
    } else {
        found = std::move(converter);
    }
}

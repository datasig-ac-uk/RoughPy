//
// Created by sam on 13/03/23.
//

#include "RationalType.h"

#include <algorithm>
#include <ostream>
#include <utility>

using namespace rpy;
using namespace rpy::scalars;

std::unique_ptr<RandomGenerator> RationalType::get_mt19937_generator(const ScalarType* type, Slice<uint64_t> seed) {
    return nullptr;
}
std::unique_ptr<RandomGenerator> RationalType::get_pcg_generator(const ScalarType* type, Slice<uint64_t> seed) {
    return std::unique_ptr<RandomGenerator>();
}

RationalType::RationalType()
    : ScalarType({}) {}
ScalarPointer RationalType::allocate(std::size_t count) const {
    if (count == 1) {
        return ScalarPointer(this, new rational_scalar_type, ScalarPointer::IsMutable);
    } else {
        return ScalarPointer(this, new rational_scalar_type[count], ScalarPointer::IsMutable);
    }
}
void RationalType::free(ScalarPointer pointer, std::size_t count) const {
    if (!pointer.is_null()) {
        if (count == 1) {
            delete pointer.template raw_cast<rational_scalar_type>();
        } else {
            delete[] pointer.template raw_cast<rational_scalar_type>();
        }
    }
}

RationalType::scalar_type RationalType::try_convert(ScalarPointer other) const {
    if (other.is_null()) {
        return scalar_type(0);
    }
    if (other.type() == this) {
        return *other.template raw_cast<const scalar_type>();
    }

    const ScalarType *type = other.type();
    if (type == nullptr) {
        throw std::runtime_error("null type for non-zero value");
    }

    auto cv = get_conversion(type->id(), this->id());
    if (cv) {
        scalar_type result;
        ScalarPointer result_ptr{this, &result};
        cv(result_ptr, other, 1);
        return result;
    }

    throw std::runtime_error("could not convert " + type->info().name + " to scalar type " + info().name);
}

void RationalType::convert_copy(ScalarPointer dst, ScalarPointer src, dimn_t count) const {
    if (src.type() == nullptr) {
        throw std::invalid_argument("source type cannot be null");
    }
    convert_copy(dst, src.ptr(), count, src.type()->id());
}
void RationalType::convert_copy(void *out, const void *in, std::size_t count, BasicScalarInfo info) const {
}
void RationalType::convert_copy(void *out, ScalarPointer in, std::size_t count) const {
    assert(out != nullptr);
    assert(!in.is_null());
    const auto *type = in.type();

    if (type == nullptr) {
        throw std::runtime_error("null type for non-zero value");
    }

    if (type == this) {
        const auto *in_begin = in.template raw_cast<const scalar_type>();
        const auto *in_end = in_begin + count;
        std::copy(in_begin, in_end, static_cast<scalar_type *>(out));
    } else {
        auto cv = get_conversion(type->id(), this->id());
        ScalarPointer out_ptr{this, out};

        cv(out_ptr, in, count);
    }
}

void RationalType::convert_copy(ScalarPointer out, const void *in, std::size_t count, const std::string &type_id) const {
    if (type_id == "f64") {
        return convert_copy_basic<double>(out, in, count);
    } else if (type_id == "f32") {
        return convert_copy_basic<float>(out, in, count);
    } else if (type_id == "i32") {
        return convert_copy_basic<int>(out, in, count);
    } else if (type_id == "u32") {
        return convert_copy_basic<unsigned int>(out, in, count);
    } else if (type_id == "i64") {
        return convert_copy_basic<long long>(out, in, count);
    } else if (type_id == "u64") {
        return convert_copy_basic<unsigned long long>(out, in, count);
    } else if (type_id == "isize") {
        return convert_copy_basic<std::ptrdiff_t>(out, in, count);
    } else if (type_id == "usize") {
        return convert_copy_basic<std::size_t>(out, in, count);
    } else if (type_id == "i16") {
        return convert_copy_basic<short>(out, in, count);
    } else if (type_id == "u16") {
        return convert_copy_basic<unsigned short>(out, in, count);
    } else if (type_id == "i8") {
        return convert_copy_basic<char>(out, in, count);
    } else if (type_id == "u8") {
        return convert_copy_basic<unsigned char>(out, in, count);
    }

    // If we're here, then it is a non-standard type
    const auto &conversion = get_conversion(type_id, this->id());
    conversion(out, {nullptr, in}, count);
}
scalar_t RationalType::to_scalar_t(ScalarPointer arg) const {
    return static_cast<scalar_t>(*arg.raw_cast<const scalar_type *>());
}
void RationalType::assign(ScalarPointer target, long long int numerator, long long int denominator) const {
    *target.raw_cast<scalar_type *>() = scalar_type(numerator) / denominator;
}
Scalar RationalType::uminus(ScalarPointer arg) const {
    return Scalar(this, -try_convert(arg));
}
void RationalType::add_inplace(ScalarPointer lhs, ScalarPointer rhs) const {
    assert(lhs);
    auto *ptr = lhs.raw_cast<scalar_type *>();
    *ptr += try_convert(rhs);
}
void RationalType::sub_inplace(ScalarPointer lhs, ScalarPointer rhs) const {
    assert(lhs);
    auto *ptr = lhs.raw_cast<scalar_type *>();
    *ptr -= try_convert(rhs);
}
void RationalType::mul_inplace(ScalarPointer lhs, ScalarPointer rhs) const {
    assert(lhs);
    auto *ptr = lhs.raw_cast<scalar_type *>();
    *ptr *= try_convert(rhs);
}
void RationalType::div_inplace(ScalarPointer lhs, ScalarPointer rhs) const {
    assert(lhs);
    auto *ptr = lhs.raw_cast<scalar_type *>();
    if (rhs.is_null()) {
        throw std::runtime_error("division by zero");
    }

    auto crhs = try_convert(rhs);

    if (crhs == scalar_type(0)) {
        throw std::runtime_error("division by zero");
    }

    *ptr /= crhs;
}
bool RationalType::are_equal(ScalarPointer lhs, ScalarPointer rhs) const noexcept {
    return *lhs.raw_cast<const scalar_type *>() == try_convert(rhs);
}
Scalar RationalType::from(long long int numerator, long long int denominator) const {
    return Scalar(this, scalar_type(numerator) / denominator);
}
void RationalType::convert_fill(ScalarPointer out, ScalarPointer in, dimn_t count, const std::string &id) const {
    ScalarType::convert_fill(out, in, count, id);
}
Scalar RationalType::one() const {
    return Scalar(this, scalar_type(1));
}
Scalar RationalType::mone() const {
    return Scalar(this, scalar_type(-1));
}
Scalar RationalType::zero() const {
    return Scalar(this, scalar_type(0));
}
Scalar RationalType::copy(ScalarPointer arg) const {
    return Scalar(this, try_convert(arg));
}
Scalar RationalType::add(ScalarPointer lhs, ScalarPointer rhs) const {
    if (!lhs) {
        return copy(rhs);
    }
    return Scalar(this, *lhs.raw_cast<const scalar_type *>() + try_convert(rhs));
}
Scalar RationalType::sub(ScalarPointer lhs, ScalarPointer rhs) const {
    if (!lhs) {
        return uminus(rhs);
    }
    return Scalar(this, *lhs.raw_cast<const scalar_type *>() - try_convert(rhs));
}
Scalar RationalType::mul(ScalarPointer lhs, ScalarPointer rhs) const {
    if (!lhs) {
        return zero();
    }
    return Scalar(this, *lhs.raw_cast<const scalar_type *>() * try_convert(rhs));
}
Scalar RationalType::div(ScalarPointer lhs, ScalarPointer rhs) const {
    if (!lhs) {
        return zero();
    }
    if (rhs.is_null()) {
        throw std::runtime_error("division by zero");
    }

    auto crhs = try_convert(rhs);

    if (crhs == scalar_type(0)) {
        throw std::runtime_error("division by zero");
    }

    return Scalar(this, static_cast<scalar_type>(*lhs.raw_cast<const scalar_type *>() / crhs));
}
bool RationalType::is_zero(ScalarPointer arg) const {
    return !static_cast<bool>(arg) || *arg.raw_cast<const scalar_type *>() == scalar_type(0);
}
void RationalType::print(ScalarPointer arg, std::ostream &os) const {
    if (!arg) {
        os << 0.0;
    } else {
        os << *arg.raw_cast<const scalar_type *>();
    }
}
std::unique_ptr<RandomGenerator> RationalType::get_rng(const std::string &bit_generator, Slice<uint64_t> seed) const {
    ScalarType::get_rng(bit_generator, seed);
    RPY_UNREACHABLE();
}

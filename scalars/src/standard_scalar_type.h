//
// Created by user on 02/03/23.
//

#ifndef ROUGHPY_SCALARS_SRC_STANDARD_SCALAR_TYPE_H
#define ROUGHPY_SCALARS_SRC_STANDARD_SCALAR_TYPE_H

#include "scalar_type.h"
#include "scalar.h"

#include <ostream>
#include <limits>
#include <utility>

namespace rpy  {
namespace scalars {

template <typename T>
constexpr std::uint8_t sizeof_bits() noexcept {
    return static_cast<std::uint8_t>(std::min(
        static_cast<std::size_t>(std::numeric_limits<std::uint8_t>::max() / 8),
        sizeof(T)
        ) * 8U);
}

template <typename ScalarImpl>
class StandardScalarType : public ScalarType {
public:
    explicit StandardScalarType(std::string id, std::string name)
        : ScalarType({{2U, sizeof_bits<ScalarImpl>(), 1U},
                      {ScalarDeviceType::CPU, 0},
                      std::move(name), std::move(id),
                      sizeof(ScalarImpl), alignof(ScalarImpl)}) {}


    Scalar from(long long int numerator, long long int denominator) const override {
        return Scalar(this, ScalarImpl(numerator) / denominator);
    }

    ScalarPointer allocate(dimn_t size) const override {
        if (size == 1) {
            return ScalarPointer(this, new ScalarImpl);
        } else {
            return ScalarPointer(this, new ScalarImpl[size]);
        }
    }
    void free(ScalarPointer pointer, dimn_t size) const override {
        if (!pointer.is_null()) {
            if (size == 1) {
                delete pointer.template raw_cast<ScalarImpl>();
            } else {
                delete[] pointer.template raw_cast<ScalarImpl>();
            }
        }
    }

protected:
    ScalarImpl try_convert(ScalarPointer other) const {
        if (other.is_null()) {
            return ScalarImpl(0);
        }
        if (other.type() == this) {
            return *other.template raw_cast<const ScalarImpl>();
        }

        const ScalarType *type = other.type();
        if (type == nullptr) {
            throw std::runtime_error("null type for non-zero value");
        }

        auto cv = get_conversion(type->id(), this->id());
        if (cv) {
            ScalarImpl result;
            ScalarPointer result_ptr{this, &result};
            cv(result_ptr, other, 1);
            return result;
        }

        throw std::runtime_error("could not convert " + type->info().name + " to scalar type " + info().name);
    }

public:
    void convert_copy(void *out, ScalarPointer in, dimn_t count) const override {
        assert(out != nullptr);
        assert(!in.is_null());
        const auto *type = in.type();

        if (type == nullptr) {
            throw std::runtime_error("null type for non-zero value");
        }

        if (type == this) {
            const auto *in_begin = in.template raw_cast<const ScalarImpl>();
            const auto *in_end = in_begin + count;
            std::copy(in_begin, in_end, static_cast<ScalarImpl *>(out));
        } else {
            auto cv = get_conversion(type->id(), this->id());
            ScalarPointer out_ptr{this, out};

            cv(out_ptr, in, count);
        }
    }

private:
    template <typename Basic>
    void convert_copy_basic(ScalarPointer &out,
                            const void *in,
                            dimn_t count) const noexcept {
        const auto *iptr = static_cast<const Basic *>(in);
        auto *optr = static_cast<ScalarImpl *>(out.ptr());

        for (dimn_t i = 0; i < count; ++i, ++iptr, ++optr) {
            ::new (optr) ScalarImpl(*iptr);
        }
    }

public:
    void convert_copy(ScalarPointer out,
                      const void *in,
                      dimn_t count,
                      const std::string &type_id) const override {
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
        const auto& conversion = get_conversion(type_id, this->id());
        conversion(out, {nullptr, in}, count);


    }

    void convert_copy(ScalarPointer dst, ScalarPointer src, dimn_t count) const override {
        if (src.type() == nullptr) {
            throw std::invalid_argument("source type cannot be null");
        }
        convert_copy(dst, src.ptr(), count, src.type()->id());
    }
    void convert_copy(void *out, const void *in, std::size_t count, BasicScalarInfo info) const override {
        //TODO: Implement me
    }
    void assign(ScalarPointer target, long long int numerator, long long int denominator) const override {
        *target.raw_cast<ScalarImpl*>() = ScalarImpl(numerator) / denominator;
    }
    scalar_t to_scalar_t(ScalarPointer arg) const override {
        return static_cast<scalar_t>(*arg.raw_cast<const ScalarImpl*>());
    }

    Scalar copy(ScalarPointer arg) const override {
        return Scalar(this, try_convert(arg));
    }
    Scalar uminus(ScalarPointer arg) const override {
        return Scalar(this, -try_convert(arg));
    }
    Scalar add(ScalarPointer lhs, ScalarPointer rhs) const override {
        if (!lhs) {
            return copy(rhs);
        }
        return Scalar(this, *lhs.raw_cast<const ScalarImpl*>() + try_convert(rhs));
    }
    Scalar sub(ScalarPointer lhs, ScalarPointer rhs) const override {
        if (!lhs) {
            return uminus(rhs);
        }
        return Scalar(this, *lhs.raw_cast<const ScalarImpl*>() - try_convert(rhs));
    }
    Scalar mul(ScalarPointer lhs, ScalarPointer rhs) const override {
        if (!lhs) {
            return zero();
        }
        return Scalar(this, *lhs.raw_cast<const ScalarImpl*>() * try_convert(rhs));
    }
    Scalar div(ScalarPointer lhs, ScalarPointer rhs) const override {
        if (!lhs) {
            return zero();
        }
        if (rhs.is_null()) {
            throw std::runtime_error("division by zero");
        }
        return Scalar(this, *lhs.raw_cast<const ScalarImpl*>() / try_convert(rhs));
    }
    bool are_equal(ScalarPointer lhs, ScalarPointer rhs) const noexcept override {
        return *lhs.raw_cast<const ScalarImpl*>() == try_convert(rhs);
    }

    Scalar one() const override {
        return Scalar(this, ScalarImpl(1));
    }
    Scalar mone() const override {
        return Scalar(this, ScalarImpl(-1));
    }
    Scalar zero() const override {
        return Scalar(this, ScalarImpl(0));
    }
    void add_inplace(ScalarPointer lhs, ScalarPointer rhs) const override {
        assert(lhs);
        auto *ptr = lhs.raw_cast<ScalarImpl*>();
        *ptr += try_convert(rhs);
    }
    void sub_inplace(ScalarPointer lhs, ScalarPointer rhs) const override {
        assert(lhs);
        auto *ptr = lhs.raw_cast<ScalarImpl *>();
        *ptr -= try_convert(rhs);
    }
    void mul_inplace(ScalarPointer lhs, ScalarPointer rhs) const override {
        assert(lhs);
        auto *ptr = lhs.raw_cast<ScalarImpl *>();
        *ptr *= try_convert(rhs);
    }
    void div_inplace(ScalarPointer lhs, ScalarPointer rhs) const override {
        assert(lhs);
        auto *ptr = lhs.raw_cast<ScalarImpl *>();
        if (rhs.is_null()) {
            throw std::invalid_argument("division by zero");
        }
        *ptr /= try_convert(rhs);
    }
    bool is_zero(ScalarPointer arg) const override {
        return !static_cast<bool>(arg) || *arg.raw_cast<const ScalarImpl *>() == ScalarImpl(0);
    }
    void print(ScalarPointer arg, std::ostream &os) const override {
        if (!arg) {
            os << 0.0;
        } else {
            os << *arg.raw_cast<const ScalarImpl*>();
        }
    }
};

}
}

#endif//ROUGHPY_SCALARS_SRC_STANDARD_SCALAR_TYPE_H

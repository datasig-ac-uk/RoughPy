//
// Created by sam on 13/03/23.
//

#ifndef ROUGHPY_RATIONALTYPE_H
#define ROUGHPY_RATIONALTYPE_H

#include "rational_type.h"
#include "standard_scalar_type.h"

namespace rpy {
namespace scalars {

class RationalType : public ScalarType {
    using scalar_type = rational_scalar_type;

    using rng_getter = std::unique_ptr<RandomGenerator> (*)(const ScalarType* type, Slice<uint64_t>);

    static std::unique_ptr<RandomGenerator> get_mt19937_generator(const ScalarType* type, Slice<uint64_t> seed);
    static std::unique_ptr<RandomGenerator> get_pcg_generator(const ScalarType* type, Slice<uint64_t> seed);

    std::unordered_map<std::string, rng_getter> m_rng_getters {
        {"mt19937", &get_mt19937_generator},
        {"pcg", &get_pcg_generator}
    };

public:

    RationalType();

    ScalarPointer allocate(std::size_t count) const override;
    void free(ScalarPointer pointer, std::size_t count) const override;

protected:

    scalar_type try_convert(ScalarPointer other) const;

public:
    void convert_copy(ScalarPointer dst, ScalarPointer src, dimn_t count) const override;
    void convert_copy(void *out, const void *in, std::size_t count, BasicScalarInfo info) const override;
    void convert_copy(void *out, ScalarPointer in, std::size_t count) const override;

private:
    template <typename Basic>
    void convert_copy_basic(ScalarPointer &out,
                            const void *in,
                            dimn_t count) const noexcept {
        const auto *iptr = static_cast<const Basic *>(in);
        auto *optr = static_cast<scalar_type *>(out.ptr());

        for (dimn_t i = 0; i < count; ++i, ++iptr, ++optr) {
            ::new (optr) scalar_type(*iptr);
        }
    }

public:
    void convert_copy(ScalarPointer out, const void *in, std::size_t count, const std::string &id) const override;
    scalar_t to_scalar_t(ScalarPointer arg) const override;
    void assign(ScalarPointer target, long long int numerator, long long int denominator) const override;
    Scalar uminus(ScalarPointer arg) const override;
    void add_inplace(ScalarPointer lhs, ScalarPointer rhs) const override;
    void sub_inplace(ScalarPointer lhs, ScalarPointer rhs) const override;
    void mul_inplace(ScalarPointer lhs, ScalarPointer rhs) const override;
    void div_inplace(ScalarPointer lhs, ScalarPointer rhs) const override;
    bool are_equal(ScalarPointer lhs, ScalarPointer rhs) const noexcept override;

    Scalar from(long long int numerator, long long int denominator) const override;
    void convert_fill(ScalarPointer out, ScalarPointer in, dimn_t count, const std::string &id) const override;
    Scalar one() const override;
    Scalar mone() const override;
    Scalar zero() const override;
    Scalar copy(ScalarPointer source) const override;
    Scalar add(ScalarPointer lhs, ScalarPointer rhs) const override;
    Scalar sub(ScalarPointer lhs, ScalarPointer rhs) const override;
    Scalar mul(ScalarPointer lhs, ScalarPointer rhs) const override;
    Scalar div(ScalarPointer lhs, ScalarPointer rhs) const override;
    bool is_zero(ScalarPointer arg) const override;
    void print(ScalarPointer arg, std::ostream &os) const override;
    std::unique_ptr<RandomGenerator> get_rng(const std::string &bit_generator, Slice<uint64_t> seed) const override;
};

}// namespace scalars
}// namespace rpy

#endif//ROUGHPY_RATIONALTYPE_H

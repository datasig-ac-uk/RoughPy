#ifndef ROUGHPY_COMPUTE_TESTING_TENSOR_HELPERS_HPP
#define ROUGHPY_COMPUTE_TESTING_TENSOR_HELPERS_HPP

#include <vector>
#include <cmath>


#include <libalgebra_lite/polynomial.h>
#include <libalgebra_lite/coefficients.h>

#include "roughpy_compute/common/basis.hpp"

namespace rpy::compute::testing {


struct PolynomialTensorHelper
{
    using Scalar = typename lal::polynomial_ring::scalar_type;
    using Monomial = typename Scalar::key_type;
    using Indeterminant = typename Monomial::letter_type;
    using Rational = typename Scalar::scalar_type;

    using Basis = TensorBasis<>;

    using Degree = typename Basis::Degree;

    [[nodiscard]]
    static Monomial make_monomial(char letter, int32_t pos)
    {
        return Monomial(Indeterminant(letter, pos));
    }

    [[nodiscard]]
    static Scalar make_scalar(std::initializer_list<std::tuple<std::initializer_list<std::pair<char, size_t>>, int32_t, int32_t>> coeffs)
    {
        Scalar result;

        for (auto&& [markers, num, denom] : coeffs) {
            Monomial mon;
            for (auto&& [marker, pos] : markers) {
                mon *= make_monomial(marker, pos);
            }

            result[mon] = Rational(num, denom);
        }

        return result;
    }

    template <typename F>
    static void for_each_word(Basis const& basis, F&& fn)
    {
        std::vector<size_t> word;
        word.reserve(basis.depth);

        fn(word);
        for (Degree deg = 1; deg <= basis.depth; ++deg) {
            word.clear();
            word.resize(deg, 1);

            auto advance = [&word, &deg, &basis] {
                size_t carry = 1;
                for (Degree pos = deg - 1; pos >= 0; --pos) {
                    word[pos] += carry;
                    if (word[pos] == static_cast<size_t>(basis.width + 1)) {
                        carry = 1;
                        word[pos] = 1;
                    } else {
                        break;
                    }
                }
            };

            for (auto i = basis.degree_begin[deg]; i < basis.degree_begin[deg +
                     1]; ++i, advance()) { fn(word); }
        }
    }


    struct WordToIdxFunction
    {
        size_t base;

        explicit WordToIdxFunction(size_t base) : base(base) {}

        size_t operator()(size_t const* letters, size_t size) const
        {
            size_t idx = 0;
            for (size_t i = 0; i < size; ++i) {
                idx *= base;
                idx += letters[i];
            }
            return idx;
        }

        template <typename LetterVec>
        size_t operator()(LetterVec const& letters) const
        {
            return (*this)(letters.data(), letters.size());
        }

        size_t operator()(size_t const* letters, size_t size, size_t mask, size_t mask_use=1) const
        {
            size_t idx = 0;
            for (size_t i = 0; i < size; ++i) {
                if (((mask >> i) & size_t{1}) == mask_use) {
                    idx *= base;
                    idx += letters[i];
                }
            }
            return idx;
        }

        template <typename LetterVec>
        size_t operator()(LetterVec const& letters, size_t mask, size_t mask_use=1) const
        {
            return (*this)(letters.data(), letters.size(), mask, mask_use);
        }
    };


    [[nodiscard]]
    static auto word_to_idx_fn(Basis const& basis)
    {
        const auto widthlog10 = std::ceil(
            std::log10(static_cast<double>(basis.width)));
        const auto base = static_cast<size_t>(std::pow(10, widthlog10));

        return WordToIdxFunction(base);
    }


    template <typename ToIdx>
    static std::vector<Scalar> make_tensor(char marker,
                                           Basis const& basis,
                                           ToIdx const& to_idx)
    {
        std::vector<Scalar> result;
        result.reserve(basis.size());

        for_each_word(basis,
                      [&to_idx, &marker, &result](auto const& word) {
                          result.emplace_back(make_scalar({
                                  {
                                          {{marker, to_idx(word)}},
                                          1, 1
                                  }
                          }));
                      });

        return result;
    }
};


}

#endif //ROUGHPY_COMPUTE_TESTING_TENSOR_HELPERS_HPP
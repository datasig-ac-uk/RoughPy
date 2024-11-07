//
// Created by user on 28/08/22.
//

#ifndef LIBALGEBRA_LITE_POLYNOMIAL_BASIS_H
#define LIBALGEBRA_LITE_POLYNOMIAL_BASIS_H

#include "implementation_types.h"

#include <functional>
#include <numeric>
#include <iosfwd>

#include <boost/container/small_vector.hpp>
#include <boost/container/flat_map.hpp>
#include <boost/container_hash/hash.hpp>

#include "libalgebra_lite_export.h"
#include "packed_integer.h"
#include "registry.h"

namespace lal {


class LIBALGEBRA_LITE_EXPORT monomial
{
public:
    using letter_type = dtl::packed_integer<dimn_t, char>;

private:
    using small_vec = boost::container::small_vector<std::pair<letter_type, deg_t>, 1>;
    using map_type = boost::container::flat_map<letter_type, deg_t, std::less<>, small_vec>;
//    using map_type = std::map<letter_type, deg_t>;

    map_type m_data;

    template <typename Coeff>
    static typename Coeff::scalar_type
    power(typename Coeff::scalar_type arg, deg_t exponent) noexcept
    {
        if (exponent == 0) {
            return Coeff::one();
        }
        if (exponent == 1) {
            return arg;
        }
        auto result1 = power<Coeff>(arg, exponent/2);
        auto result2 = Coeff::mul(result1, result1);
        return (exponent % 2==0) ? result2 : Coeff::mul(arg, result2);
    }

public:

    using iterator = typename map_type::iterator;
    using const_iterator = typename map_type::const_iterator;

    monomial() : m_data() {}

    explicit monomial(letter_type let, deg_t power=1)
    {
        assert(power > 0);
        m_data[let] = power;
    }

    template <typename MapType>
    explicit monomial(const MapType& arg) : m_data(arg.begin(), arg.end())
    {}

    template <typename InputIt>
    explicit monomial(InputIt begin, InputIt end) : m_data(begin, end)
    {}

    deg_t degree() const noexcept;
    deg_t type() const noexcept { return m_data.size(); }

    deg_t operator[](letter_type let) const noexcept;
    deg_t& operator[](letter_type let) noexcept { return m_data[let]; }

    iterator begin() noexcept { return m_data.begin(); }
    iterator end() noexcept { return m_data.end(); }
    const_iterator begin() const noexcept { return m_data.begin(); }
    const_iterator end() const noexcept { return m_data.end(); }

    template <typename Coefficients, typename MapType>
    typename Coefficients::scalar_type eval(const MapType& arg) const noexcept
    {
        auto result = Coefficients::zero();

        for (const auto& item : m_data) {
            Coefficients::add_inplace(result,
                    power<Coefficients>(arg[item.first], item.second));
        }
        return result;
    }

    bool operator==(const monomial& other) const noexcept
    {
        return degree() == other.degree() && m_data == other.m_data;
    }

    bool operator<(const monomial& other) const noexcept
    {
        auto ldegree = degree();
        auto rdegree = other.degree();
        return (ldegree < rdegree) || (ldegree == rdegree && (m_data < other.m_data));
    }

    monomial& operator*=(const monomial& rhs);

    friend std::size_t hash_value(const monomial& mon) noexcept
    {
        boost::hash<map_type> hasher;
        return hasher(mon.m_data);
    }

};

LIBALGEBRA_LITE_EXPORT std::ostream& operator<<(std::ostream& os, const monomial& arg);

LIBALGEBRA_LITE_EXPORT monomial operator*(const monomial& lhs, const monomial& rhs);

struct LIBALGEBRA_LITE_EXPORT polynomial_basis
{
    using letter_type = dtl::packed_integer<dimn_t, char>;
    using key_type = monomial;

    static key_type key_of_letter(letter_type letter)
    {
        return key_type(letter);
    }

    static deg_t degree(const key_type& key)
    {
        return key.degree();
    }

    std::ostream& print_key(std::ostream& os, const key_type& key) const;

};


template <>
class LIBALGEBRA_LITE_EXPORT basis_registry<polynomial_basis>
{
public:

    static basis_pointer<polynomial_basis> get();
    static basis_pointer<polynomial_basis> get(const polynomial_basis&) { return get(); }

};

} // namespace lal

#endif //LIBALGEBRA_LITE_POLYNOMIAL_BASIS_H

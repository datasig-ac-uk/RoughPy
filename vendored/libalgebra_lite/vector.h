//
// Created by user on 08/08/22.
//

#ifndef LIBALGEBRA_LITE_VECTOR_H
#define LIBALGEBRA_LITE_VECTOR_H

#include "implementation_types.h"

#include <ostream>
#include <memory>

#include "basis.h"
#include "vector_traits.h"
#include "coefficients.h"
#include "basis_traits.h"
#include "registry.h"
#include "vector_base.h"

namespace lal {

namespace dtl {

template <typename VectorType>
struct storage_base {
    using vector_type = VectorType;
    using vect_traits = vector_traits<VectorType>;

    using basis_type = typename vect_traits::basis_type;
    using registry = basis_registry<basis_type>;
    using coefficient_ring = typename vect_traits::coefficient_ring;

    using basis_traits = basis_trait<basis_type>;
    using coeff_traits = coefficient_trait<coefficient_ring>;

    using basis_pointer = lal::basis_pointer<basis_type>;
    using key_type = typename basis_traits::key_type;
    using scalar_type = typename coeff_traits::scalar_type;
    using rational_type = typename coeff_traits::rational_type;

    using iterator = typename vector_type::iterator;
    using const_iterator = typename vector_type::const_iterator;
    using reference = typename vector_type::reference;
    using const_reference = typename vector_type::const_reference;

    static_assert(std::is_base_of<lal::vectors::vector_base<basis_type, coefficient_ring>, VectorType>::value,
                  "vector type must derive from the vector_base class");

};

template <typename VectorType>
class standard_storage : public storage_base<VectorType> {
public:
    using base = storage_base<VectorType>;

    using typename base::vector_type;

    using typename base::vect_traits;
    using typename base::basis_traits;
    using typename base::coeff_traits;
    using typename base::registry;
    using typename base::basis_type;
    using typename base::key_type;
    using typename base::coefficient_ring;
    using typename base::scalar_type;
    using typename base::rational_type;
    using typename base::iterator;
    using typename base::const_iterator;
    using typename base::reference;
    using typename base::const_reference;

    using typename base::basis_pointer;


private:
    vector_type m_instance;

protected:
    const vector_type &instance() const noexcept { return m_instance; }
    vector_type &instance() noexcept { return m_instance; }

public:

    standard_storage() : m_instance(registry::get()) {}

    standard_storage(const standard_storage& other)
        : m_instance(other.m_instance)
    {}

    standard_storage(standard_storage&& other) noexcept
        : m_instance(std::move(other.m_instance))
    {}

    explicit standard_storage(basis_pointer basis) : m_instance(basis) {}

    explicit standard_storage(vector_type &&data)
        :  m_instance(std::move(data)) {
    }

    template <typename... VArgs>
    explicit standard_storage(basis_pointer basis, VArgs&&... args)
        : m_instance(basis, std::forward<VArgs>(args)...)
    {}

    standard_storage& operator=(const standard_storage& other)
    {
        if (&other != this) {
            m_instance = other.m_instance;
        }
        return *this;
    }

    standard_storage& operator=(standard_storage&& other) noexcept {
        if (&other != this) {
            m_instance = std::move(other.m_instance);
        }
        return *this;
    }


};

} // namespace dtl

#define LAL_VECTYPE_AND_STORAGE_TEMPLATE_ARGS(VT, SM, EA) \
    template <typename, typename> class VT, template <typename> class SM
#define LAL_VECTOR_TEMPLATE_ARGS(B, C, VT, SM, EA) \
    typename B, typename C, LAL_VECTYPE_AND_STORAGE_TEMPLATE_ARGS((SM), (EA))

template <typename Basis,
    typename Coefficients,
    template <typename, typename> class VectorType,
    template <typename> class StorageModel=dtl::standard_storage>
class vector : protected StorageModel<VectorType<Basis, Coefficients>> {
    using base_type = StorageModel<VectorType<Basis, Coefficients>>;

public:
    using typename base_type::vector_type;
    using typename base_type::basis_type;
    using typename base_type::basis_pointer;
    using typename base_type::key_type;
    using typename base_type::coefficient_ring;
    using typename base_type::scalar_type;
    using typename base_type::rational_type;

    using typename base_type::registry;
    using typename base_type::iterator;
    using typename base_type::const_iterator;
    using typename base_type::reference;
    using typename base_type::const_reference;

protected:

    vector(vector_type &&arg)
        : base_type(std::move(arg)) {}

public:

    vector() : base_type() {}

    vector(const vector& other) : base_type(other)
    {
    }

    vector(vector&& other) noexcept : base_type(std::move(other))
    {}

    template <typename Scalar>
    explicit vector(key_type k, Scalar s) : base_type(k, s)
    {}

//    template <typename Key, typename Scalar>
//    explicit vector(Key k, Scalar s) : base_type(p_basis,
//                                                 key_type(k),
//                                                 scalar_type(s)) {
//    }

    template <typename Key, typename Scalar>
    explicit vector(basis_pointer basis, Key key, Scalar s)
        : base_type(std::move(basis), key_type(key), scalar_type(s)) {}

    explicit vector(basis_pointer basis) : base_type(std::move(basis)) {}

    vector(basis_pointer basis, std::initializer_list<scalar_type> args)
        : base_type(vector_type(basis, args)) {}

    explicit vector(const vector_type &arg)
        : base_type(vector_type(arg)) {
    }


    vector& operator=(const vector&) = default;
    vector& operator=(vector&&) noexcept = default;


    vector clone() const {
        return vector(*this);
    }

    vector create_alike() const { return vector(get_basis()); }

    vector_type &base_vector() noexcept { return base_type::instance(); }
    const vector_type &base_vector() const noexcept { return base_type::instance(); }

    dimn_t size() const noexcept {
        return base_type::instance().size();
    }

    dimn_t dimension() const noexcept {
        return base_type::instance().dimension();
    }

    deg_t degree() const noexcept { return base_type::instance().degree(); }
    bool empty() const noexcept {
        return base_type::instance().empty();
    }

    const basis_type& basis() const noexcept { return base_type::instance().basis(); }
    basis_pointer get_basis() const noexcept { return base_type::instance().get_basis(); }

    template <typename KeyType>
    const_reference operator[](const KeyType &key) const {
        return base_type::instance()[key_type(key)];
    }

    template <typename KeyType>
    reference operator[](const KeyType &key) {
        return base_type::instance()[key_type(key)];
    }

    void clear() {
        base_type::instance().clear();
    }

    iterator begin() noexcept {
        return base_type::instance().begin();
    }

    iterator end() noexcept {
        return base_type::instance().end();
    }
    const_iterator begin() const noexcept {
        return base_type::instance().cbegin();
    }
    const_iterator end() const noexcept {
        return base_type::instance().cend();
    }

private:

    template <typename, typename, typename, typename=void>
    struct has_iterator_inplace_binop : std::false_type {};

    template <typename V, typename F, typename I>
    struct has_iterator_inplace_binop<V, F, I, std::void_t<
        decltype(V::template inplace_binop(
            std::declval<I>(),
            std::declval<I>(),
            std::declval<F>()
        ))>>
        : std::true_type {
    };

public:

//    template <typename Iter, typename C>
//    std::enable_if_t<
//        has_iterator_inplace_binop<
//            vector_type,
//            decltype(coefficient_ring::template add_inplace<>),
//            Iter
//        >::value,
//        vector &>
//    add_inplace(Iter begin, Iter end) {
//        return base_type::instance()
//            .inplace_binop(begin, end, coefficient_ring::add_inplace);
//    }
//
//    template <typename Iter>
//    std::enable_if_t<
//        has_iterator_inplace_binop<
//            vector_type,
//            decltype(coefficient_ring::template add_inplace<>),
//            Iter>::value,
//        vector &>
//    sub_inplace(Iter begin, Iter end) {
//        return base_type::instance()
//            .inplace_binop(begin, end, coefficient_ring::sub_inplace);
//    }

    template <typename Iter>
//    std::enable_if_t<
//        !has_iterator_inplace_binop<
//            vector_type,
//            decltype(coefficient_ring::template add_inplace<>),
//            Iter>::value,
//        vector &>
    vector&
    add_inplace(Iter begin, Iter end) {
        const auto &self = base_type::instance();
        for (auto it = begin; it != end; ++it) {
            self[it->first] += it->second;
        }
        return *this;
    }

    template <typename Iter>
//    std::enable_if_t<
//        !has_iterator_inplace_binop<
//            vector_type,
//            decltype(coefficient_ring::template sub_inplace<>),
//            Iter>::value,
//        vector &>
    vector&
    sub_inplace(Iter begin, Iter end) {
        const auto &self = base_type::instance();
        for (auto it = begin; it != end; ++it) {
            self[it->first] -= it->second;
        }
    }

    template <typename Key, typename Scal>
    std::enable_if_t<std::is_constructible<key_type, const Key&>::value, vector&>
    add_scal_prod(const Key &key, const Scal &scal)
    {
        base_type::instance()[key_type(key)] += scalar_type(scal);
        return *this;
    }
    template <typename Key, typename Rat>
    std::enable_if_t<!std::is_base_of<vector, Key>::value, vector&>
    add_scal_div(const Key &key, const Rat &scal)
    {
        base_type::instance()[key_type(key)] +=
            coefficient_ring::one() / rational_type(scal);
        return *this;
    }
    template <typename Key, typename Scal>
    std::enable_if_t<std::is_constructible<key_type, const Key&>::value, vector&>
    sub_scal_prod(const Key &key, const Scal &scal)
    {
        base_type::instance()[key_type(key)] -= scalar_type(scal);
        return *this;
    }
    template <typename Key, typename Rat>
    std::enable_if_t<!std::is_base_of<vector, Key>::value, vector &>
    sub_scal_div(const Key &key, const Rat &scal)
    {
        base_type::instance()[key_type(key)] -=
            coefficient_ring::one() / rational_type(scal);
        return *this;
    }

    template <typename Scal>
    vector &add_scal_prod(const vector &rhs, const Scal &scal)
    {
        auto &self = base_vector();
        scalar_type m(scal);
        self.inplace_binary_op(rhs.base_vector(), [m](scalar_type &ls, const scalar_type &rs) { ls += rs * m; });
        return *this;
    }
    template <typename Rat>
    vector &add_scal_div(const vector &rhs, const Rat &scal)
    {
        auto &self = base_vector();
        rational_type m(scal);
        self.inplace_binary_op(rhs.base_vector(), [m](scalar_type &ls, const scalar_type &rs) { ls += rs / m; });
        return *this;
    }
    template <typename Scal>
    vector &sub_scal_prod(const vector &rhs, const Scal &scal)
    {
        auto &self = base_vector();
        scalar_type m(scal);
        self.inplace_binary_op(rhs.base_vector(), [m](scalar_type &ls, const scalar_type &rs) { ls -= rs * m; });
        return *this;
    }
    template <typename Rat>
    vector &sub_scal_div(const vector &rhs, const Rat &scal)
    {
        auto &self = base_vector();
        rational_type m(scal);
        self.inplace_binary_op(rhs.base_vector(), [m](scalar_type &ls, const scalar_type &rs) { ls -= rs / m; });
        return *this;
    }

    template <template <typename, typename> class AltVecType,
        template <typename> class AltStorageModel,
        typename Scal>
    vector &add_scal_prod(
        const vector<Basis, Coefficients, AltVecType, AltStorageModel> &rhs,
        const Scal &scal
    )
    {
        const auto &self = this->instance();
        const scalar_type m(scal);
        for (auto term : rhs.instance()) {
            self[term.key()] += term.value() * m;
        }
        return *this;
    }
    template <template <typename, typename> class AltVecType,
        template <typename> class AltStorageModel,
        typename Rat>
    vector &add_scal_div(
        const vector<Basis, Coefficients, AltVecType, AltStorageModel> &rhs,
        const Rat &scal
    )
    {
        const auto &self = this->instance();
        const rational_type m(scal);
        for (auto term : rhs.instance()) {
            self[term.key()] += term.value() / m;
        }
        return *this;
    }
    template <template <typename, typename> class AltVecType,
        template <typename> class AltStorageModel,
        typename Scal>
    vector &sub_scal_prod(
        const vector<Basis, Coefficients, AltVecType, AltStorageModel> &rhs,
        const Scal &scal
    )
    {
        const auto& self = this->instance();
        const scalar_type m(scal);
        for (auto term : rhs.instance()) {
            self[term.key()] -= term.value()*m;
        }
        return *this;
    }
    template <template <typename, typename> class AltVecType,
        template <typename> class AltStorageModel,
        typename Rat>
    vector &sub_scal_div(
        const vector<Basis, Coefficients, AltVecType, AltStorageModel> &rhs,
        const Rat &scal
    )
    {
        const auto &self = this->instance();
        const rational_type m(scal);
        for (auto term : rhs.instance()) {
            self[term.key()] -= term.value() / m;
        }
        return *this;
    }

/*    template <
        typename AltBasis,
        typename AltCoeffs,
        template <typename, typename> class AltVecType,
        template <typename> class AltStorageModel,
        typename Scal>
    vector &add_scal_prod(
        const vector<AltBasis, AltCoeffs, AltVecType, AltStorageModel> &rhs,
        const Scal &scal
    )
    {
        return *this;
    }
    template <
        typename AltBasis,
        typename AltCoeffs,
        template <typename, typename> class AltVecType,
        template <typename> class AltStorageModel,
        typename Rat>
    vector &add_scal_div(
        const vector<AltBasis, AltCoeffs, AltVecType, AltStorageModel> &rhs,
        const Rat &scal
    )
    {
        return *this;
    }
    template <
        typename AltBasis,
        typename AltCoeffs,
        template <typename, typename> class AltVecType,
        template <typename> class AltStorageModel,
        typename Scal>
    vector &sub_scal_prod(
        const vector<AltBasis, AltCoeffs, AltVecType, AltStorageModel> &rhs,
        const Scal &scal
    )
    {
        return *this;
    }
    template <
        typename AltBasis,
        typename AltCoeffs,
        template <typename, typename> class AltVecType,
        template <typename> class AltStorageModel,
        typename Rat>
    vector &sub_scal_div(
        const vector<AltBasis, AltCoeffs, AltVecType, AltStorageModel> &rhs,
        const Rat &scal
    )
    {
        return *this;
    }*/

    template <typename Vector>
    friend std::enable_if_t<std::is_base_of<vector, Vector>::value, Vector>
    operator-(const Vector &arg) {
        vector result_inner(arg.instance()
                                .unary_op([](const scalar_type &s) { return -s; }));
        return Vector(std::move(result_inner));
    }

    template <typename Vector, typename Scal>
    friend std::enable_if_t<std::is_base_of<vector, Vector>::value && (
        std::is_same<Scal, scalar_type>::value || std::is_constructible<scalar_type, const Scal&>::value),
                            Vector>
    operator*(const Vector &arg, const Scal &scalar) {
        scalar_type m(scalar);
        vector result_inner(arg.instance()
                                .unary_op([m](const scalar_type &s) {
                                    return s * m;
                                }));
        return Vector(std::move(result_inner));
    }

    template <typename Vector, typename Scal>
    friend std::enable_if_t<std::is_base_of<vector, Vector>::value &&
        (std::is_same<Scal, scalar_type>::value || std::is_constructible<scalar_type, const Scal &>::value),
    Vector>
    operator*(const Scal &scalar, const Vector &arg) {
        scalar_type m(scalar);
        vector result_inner(arg.instance()
                                .unary_op([m](const scalar_type &s) {
                                    return m * s;
                                }));
        return Vector(std::move(result_inner));
    }

    template <typename Vector, typename Rat>
    friend std::enable_if_t<std::is_base_of<vector, Vector>::value, Vector>
    operator/(const Vector &arg, const Rat &scalar) {
        rational_type m(scalar);
        vector result_inner(arg.instance()
            .unary_op([m](const scalar_type &s) { return s / m; }));
        return Vector(std::move(result_inner));
    }

    template <typename LVector>
    friend std::enable_if_t<std::is_base_of<vector, LVector>::value, LVector>
    operator+(const LVector &lhs, const vector &rhs) {
        vector result_inner(lhs.instance().binary_op(rhs.instance(),
                                                                         [](const scalar_type &ls,
                                                                            const scalar_type &rs) {
                                                                             return
                                                                                 ls + rs;
                                                                         }));
        return LVector(std::move(result_inner));
    }

    template <typename LVector>
    friend std::enable_if_t<std::is_base_of<vector, LVector>::value, LVector>
    operator-(const LVector &lhs, const vector &rhs) {
        vector result_inner(lhs.instance().binary_op(rhs.instance(),
                                                     [](const scalar_type &ls, const scalar_type & rs)
                                                        { return ls - rs; }));
        return LVector(std::move(result_inner));
    }
//
//    template <typename LVector,
//        typename RCoefficients,
//        template <typename, typename> class RVecType,
//        template <typename> class RStorageModel>
//    friend std::enable_if_t<
//        std::is_base_of<vector, LVector>::value,
//        LVector
//    >
//    operator+(const LVector &lhs,
//              const vector<Basis, RCoefficients, RVecType, RStorageModel> &rhs) {
//        using rscalar_type = typename coefficient_trait<RCoefficients>::scalar_type;
//
//        return LVector(lhs.p_basis);
//
//    }

    template <typename LVector, typename Scal>
    friend std::enable_if_t<std::is_base_of<vector, LVector>::value, LVector &>
    operator*=(LVector &lhs, const Scal &scal) {
        scalar_type m(scal);
        lhs.instance().inplace_unary_op([m] (scalar_type& s) {  s *= m; });
        return lhs;
    }

    template <typename LVector, typename Rat>
    friend std::enable_if_t<std::is_base_of<vector, LVector>::value, LVector &>
    operator/=(LVector &lhs, const Rat &scal) {
        rational_type m(scal);
        lhs.instance().inplace_unary_op([m](scalar_type &s) { s /= m; });
        return lhs;
    }

    template <typename LVector>
    friend std::enable_if_t<std::is_base_of<vector, LVector>::value, LVector &>
    operator+=(LVector &lhs, const vector &rhs) {
        lhs.instance().inplace_binary_op(rhs.instance(), [](scalar_type& ls, const scalar_type& rs)
        { ls += rs; });
        return lhs;
    }

    template <typename LVector>
    friend std::enable_if_t<std::is_base_of<vector, LVector>::value, LVector &>
    operator-=(LVector &lhs, const vector &rhs) {
        lhs.instance()
            .inplace_binary_op(rhs.instance(), [](scalar_type &ls, const scalar_type &rs) { ls -= rs; });
        return lhs;
    }

    template <typename Vector>
    static Vector new_like(const Vector &arg) {
        return Vector(arg.get_basis());
    }


    friend std::ostream& operator<<(std::ostream& os, const vector& arg) {
        const auto &basis = arg.basis();
        const auto &zero = coefficient_ring::zero();
        os << "{ ";
        for (auto item : arg) {
            auto val = item.value();
            if (val != zero) {
                os << val << '(';
                basis.print_key(os, item.key());
                os << ") ";
            }
        }
        os << '}';
        return os;
    }

};

template <typename B, typename C, template <typename, typename> class VT,
    template <typename> class SM>
bool operator==(const vector<B, C, VT, SM> &lhs,
                const vector<B, C, VT, SM> &rhs) noexcept {
    return lhs.base_vector() == rhs.base_vector();
}
template <typename B, typename C, template <typename, typename> class VT,
    template <typename> class SM>
bool operator!=(const vector<B, C, VT, SM> &lhs,
                const vector<B, C, VT, SM> &rhs) noexcept {
    return !(lhs == rhs);
}
//
//template <typename Basis, typename Coeff,
//    template <typename, typename> class VectorType,
//    template <typename> class StorageModel>
//std::ostream &operator<<(std::ostream &os,
//                         const vector<Basis,
//                                      Coeff,
//                                      VectorType,
//                                      StorageModel> &vect) {
//    const auto &basis = vect.basis();
//    const auto &zero = coefficient_trait<coeff>::coefficient_ring::zero();
//    os << "{ ";
//    for (auto item : vect) {
//        auto val = item.value();
//        if (item.value() != zero) {
//            os << item.value() << '(';
//            basis.print_key(os, item.key());
//            os << ") ";
//        }
//    }
//    os << '}';
//    return os;
//}
//


} // namespace alg


#endif //LIBALGEBRA_LITE_VECTOR_H

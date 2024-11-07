//
// Created by user on 12/08/22.
//

#ifndef LIBALGEBRA_LITE_MAPS_H
#define LIBALGEBRA_LITE_MAPS_H

#include "implementation_types.h"
#include "libalgebra_lite_export.h"

#include <boost/container/small_vector.hpp>

#include "free_tensor.h"
#include "lie.h"
#include "tensor_basis.h"

namespace lal {

namespace dtl {

class generic_commutator
{
    const tensor_basis& m_basis;
    const free_tensor_multiplication& m_mul;

public:
    using key_type = typename tensor_basis::key_type;
    using pair_type = std::pair<key_type, int>;
    using tensor_type = boost::container::small_vector<pair_type, 1>;
    using ref_type = const boost::container::small_vector_base<pair_type>&;

    generic_commutator(
            const tensor_basis& basis, const free_tensor_multiplication& mul
    )
        : m_basis(basis), m_mul(mul)
    {}

    tensor_type operator()(ref_type lhs, ref_type rhs) const;
};

struct rbracketing_cache_item {
    using key_type = typename tensor_basis::key_type;
    boost::container::small_vector<std::pair<key_type, int>, 1> value;
    bool set = false;
};

class LIBALGEBRA_LITE_EXPORT maps_implementation
{
    const tensor_basis* p_tensor_basis;
    const hall_basis* p_lie_basis;
    std::shared_ptr<const lie_multiplication> p_lie_mul;
    std::shared_ptr<const free_tensor_multiplication> p_ftensor_mul;

public:
    using lkey_type = typename hall_basis::key_type;
    using tkey_type = typename tensor_basis::key_type;
    using generic_scalar_type = int;

    using lie_pair = std::pair<lkey_type, generic_scalar_type>;
    using tensor_pair = std::pair<tkey_type, generic_scalar_type>;
    using generic_lie = boost::container::small_vector<lie_pair, 1>;
    using generic_tensor = boost::container::small_vector<tensor_pair, 1>;

    using gtensor_ref = const boost::container::small_vector_base<tensor_pair>&;
    using glie_ref = const boost::container::small_vector_base<lie_pair>&;

private:
    static generic_tensor expand_letter(let_t letter);

    hall_extension<decltype(&expand_letter), generic_commutator, gtensor_ref>
            m_expand;

    mutable std::unordered_map<tkey_type, dtl::rbracketing_cache_item>
            m_rbracketing_cache;
    mutable std::recursive_mutex m_rbracketing_lock;

public:
    maps_implementation(const tensor_basis* tbasis, const hall_basis* lbasis)
        : p_tensor_basis(tbasis), p_lie_basis(lbasis),
          p_lie_mul(multiplication_registry<lie_multiplication>::get(
                  lbasis->width()
          )),
          p_ftensor_mul(
                  multiplication_registry<free_tensor_multiplication>::get(
                          tbasis->width()
                  )
          ),
          m_expand(
                  p_lie_basis->get_hall_set(), &expand_letter,
                  generic_commutator(*p_tensor_basis, *p_ftensor_mul)
          )
    {}

    glie_ref rbracketing(tkey_type tkey) const;
    gtensor_ref expand(lkey_type lkey) const;
};

}// namespace dtl

class LIBALGEBRA_LITE_EXPORT maps
{
    basis_pointer<tensor_basis> p_tensor_basis;
    basis_pointer<hall_basis> p_lie_basis;
    const dtl::maps_implementation* p_impl;

public:
    using tkey_type = typename tensor_basis::key_type;
    using lkey_type = typename hall_basis::key_type;
    using generic_scalar_type =
            typename dtl::maps_implementation::generic_scalar_type;
    using generic_lie = typename dtl::maps_implementation::generic_lie;
    using generic_tensor = typename dtl::maps_implementation::generic_tensor;
    using glie_ref = typename dtl::maps_implementation::glie_ref;
    using gtensor_ref = typename dtl::maps_implementation::gtensor_ref;

    maps(basis_pointer<tensor_basis> tbasis, basis_pointer<hall_basis> lbasis);
    maps(deg_t width, deg_t depth);
    ~maps();

    glie_ref rbracketing(tkey_type tkey) const
    {
        return p_impl->rbracketing(tkey);
    }
    gtensor_ref expand(lkey_type lkey) const { return p_impl->expand(lkey); }

    template <
            typename Coefficients,
            template <typename, typename> class VectorType,
            template <typename> class StorageModel>
    free_tensor<Coefficients, VectorType, StorageModel>
    lie_to_tensor(const lie<Coefficients, VectorType, StorageModel>& arg) const
    {
        using scalar_type =
                typename coefficient_trait<Coefficients>::scalar_type;
        if (arg.basis().width() != p_lie_basis->width()) {
            throw std::invalid_argument("mismatched width");
        }

        auto max_deg = p_lie_basis->depth();
        free_tensor<Coefficients, VectorType, StorageModel> result(
                p_tensor_basis
        );
        if (arg.basis().depth() <= max_deg) {
            for (auto outer : arg) {
                auto val = outer.value();
                for (auto inner : expand(outer.key())) {
                    assert(inner.first.degree() == outer.key().degree());
                    result.add_scal_prod(
                            inner.first, scalar_type(inner.second) * val
                    );
                }
            }
        } else {
            for (auto outer : arg) {
                auto key = outer.key();
                auto val = outer.value();
                if (p_lie_basis->degree(key) <= max_deg) {
                    for (auto inner : expand(key)) {
                        assert(outer.key().degree() == inner.first.degree());
                        result.add_scal_prod(
                                inner.first, scalar_type(inner.second) * val
                        );
                    }
                }
            }
        }
        return result;
    }

    template <
            typename Coefficients,
            template <typename, typename> class VectorType,
            template <typename> class StorageModel>
    lie<Coefficients, VectorType, StorageModel>
    tensor_to_lie(const free_tensor<Coefficients, VectorType, StorageModel>& arg
    ) const
    {
        using scalar_type = typename Coefficients::scalar_type;
        using rational_type = typename Coefficients::rational_type;

        if (arg.basis().width() != p_tensor_basis->width()) {
            throw std::invalid_argument("mismatched width");
        }
        auto max_deg = p_tensor_basis->depth();

        lie<Coefficients, VectorType, StorageModel> result(p_lie_basis);

        if (arg.basis().depth() <= max_deg) {
            for (auto&& outer : arg) {
                auto key = outer.key();
                auto deg = key.degree();
                if (deg > 0) {
                    auto val = outer.value() / rational_type(deg);
                    for (auto inner : rbracketing(key)) {
                        assert(inner.first.degree() == deg);
                        result.add_scal_prod(
                                inner.first, scalar_type(inner.second) * val
                        );
                    }
                }
            }
        } else {
            for (auto outer : arg) {
                auto key = outer.key();
                auto deg = static_cast<deg_t>(key.degree());
                if (deg > 0) {
                    auto val = outer.value() / deg;
                    if (deg <= max_deg) {
                        for (auto inner : rbracketing(key)) {
                            assert(inner.first.degree()
                                   == static_cast<dimn_t>(deg));
                            result.add_scal_prod(
                                    inner.first, scalar_type(inner.second) * val
                            );
                        }
                    }
                }
            }
        }
        return result;
    }
};

}// namespace lal

#endif// LIBALGEBRA_LITE_MAPS_H

#ifndef ROUGHPY_ALGEBRA_BASIS_IMPL_H_
#define ROUGHPY_ALGEBRA_BASIS_IMPL_H_

#include "basis.h"

#include <roughpy/config/traits.h>

#include <sstream>

namespace rpy {
namespace algebra {

template <typename RealBasis>
class BasisImplementation : public BasisInterface {
    const RealBasis *p_basis;
    using traits = basis_info<RealBasis>;

public:
    explicit BasisImplementation(const RealBasis *basis)
        : p_basis(basis) {
    }

    optional<deg_t> width() const noexcept override {
        if (p_basis != nullptr) {
            return p_basis->width();
        }
        return {};
    }
    optional<deg_t> depth() const noexcept override {
        if (p_basis != nullptr) {
            return p_basis->depth();
        }
        return {};
    }
    optional<deg_t> degree(rpy::key_type key) const noexcept override {
        if (p_basis != nullptr) {
            return traits::degree(*p_basis, key);
        }
        return {};
    }
    std::string key_to_string(const rpy::key_type key) const override {
        if (p_basis != nullptr) {
            std::stringstream ss;
            p_basis->print_key(ss, traits::convert_key(*p_basis, key));
            return ss.str();
        }
        return std::to_string(key);
    }
    dimn_t size(deg_t degree) const noexcept override {
        if (p_basis != nullptr) {
            return p_basis->size(degree);
        }
        return 0;
    }
    dimn_t start_of_degree(deg_t degree) const noexcept override {
        if (p_basis != nullptr) {
            return p_basis->start_of_degree(degree);
        }
        return 0;
    }
    optional<key_type> lparent(rpy::key_type key) const override {
        if (p_basis != nullptr) {
            return traits::convert_key(*p_basis,
                                       p_basis->lparent(traits::convert_key(*p_basis, key)));
        }
        return {};
    }
    optional<key_type> rparent(rpy::key_type key) const override {
        if (p_basis != nullptr) {
            return traits::convert_key(*p_basis,
                                       p_basis->rparent(traits::convert_key(*p_basis, key)));
        }
        return {};
    }

    key_type index_to_key(dimn_t index) const override {
        if (p_basis != nullptr) {
            return traits::convert_key(*p_basis,
                                       p_basis->index_to_key(index));
        }
        return index;
    }
    dimn_t key_to_index(rpy::key_type key) const override {
        if (p_basis != nullptr) {
            return p_basis->key_to_index(traits::convert_key(*p_basis, key));
        }
        return 0;
    }
    optional<let_t> first_letter(rpy::key_type key) const noexcept override {
        if (p_basis != nullptr) {
            return p_basis->first_letter(traits::convert_key(*p_basis, key));
        }
        return {};
    }
    optional<key_type> key_of_letter(rpy::let_t letter) const noexcept override {
        if (p_basis != nullptr) {
            return traits::convert_key(*p_basis, p_basis->key_of_letter(letter));
        }
        return letter;
    }
    bool letter(rpy::key_type key) const override {
        if (p_basis != nullptr) {
            return p_basis->letter(traits::convert_key(*p_basis, key));
        }
        return BasisInterface::letter(key);
    }
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_BASIS_IMPL_H_

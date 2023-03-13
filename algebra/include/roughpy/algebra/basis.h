#ifndef ROUGHPY_ALGEBRA_BASIS_H_
#define ROUGHPY_ALGEBRA_BASIS_H_

#include "algebra_fwd.h"

#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>


namespace rpy {
namespace algebra {

class ROUGHPY_ALGEBRA_EXPORT BasisInterface
    : public boost::intrusive_ref_counter<BasisInterface>
{
public:
    virtual ~BasisInterface() = default;

    virtual optional<deg_t> width() const noexcept;
    virtual optional<deg_t> depth() const noexcept;
    virtual optional<deg_t> degree(rpy::key_type key) const noexcept;
    virtual std::string key_to_string(const rpy::key_type key) const;
    virtual dimn_t size(deg_t degree) const noexcept;
    virtual dimn_t start_of_degree(deg_t degree) const noexcept;

    virtual optional<rpy::key_type> lparent(rpy::key_type key) const;
    virtual optional<rpy::key_type> rparent(rpy::key_type key) const;

    virtual rpy::key_type index_to_key(dimn_t index) const;
    virtual dimn_t key_to_index(rpy::key_type key) const;

    virtual optional<let_t> first_letter(rpy::key_type key) const noexcept;

    virtual optional<rpy::key_type> key_of_letter(rpy::let_t letter) const noexcept;
    virtual bool letter(rpy::key_type key) const;
};


template <typename T>
class BasisImplementation;

class ROUGHPY_ALGEBRA_EXPORT Basis {
    boost::intrusive_ptr<const BasisInterface> p_impl;

public:

    template <typename B>
    explicit Basis(const B* b) : p_impl(basis_info<B>::make(b))
    {}


    optional<deg_t> width() const noexcept;
    optional<deg_t> depth() const noexcept;
    optional<deg_t> degree(rpy::key_type key) const noexcept;
    std::string key_to_string(rpy::key_type key) const;
    dimn_t size(int deg) const noexcept;
    dimn_t start_of_degree(int degree) const noexcept;
    optional<rpy::key_type> lparent(rpy::key_type key) const;
    optional<rpy::key_type> rparent(rpy::key_type key) const;
    rpy::key_type index_to_key(dimn_t index) const;
    optional<let_t> first_letter(rpy::key_type key) const;
    optional<rpy::key_type> key_of_letter(let_t letter) const noexcept;
    bool letter(rpy::key_type key) const;

};



}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_BASIS_H_

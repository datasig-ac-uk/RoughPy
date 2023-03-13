//
// Created by user on 02/03/23.
//
#include "basis.h"

using namespace rpy;
using namespace rpy::algebra;

optional<deg_t> BasisInterface::width() const noexcept {
    return {};
}
optional<deg_t> BasisInterface::depth() const noexcept {
    return {};
}
optional<deg_t> BasisInterface::degree(rpy::key_type key) const noexcept {
    return {};
}
std::string BasisInterface::key_to_string(const rpy::key_type key) const {
    return {};
}
dimn_t BasisInterface::size(deg_t degree) const noexcept {
    return 0;
}
dimn_t BasisInterface::start_of_degree(deg_t degree) const noexcept {
    return 0;
}
optional<rpy::key_type> BasisInterface::lparent(rpy::key_type key) const {
    return {};
}
optional<rpy::key_type> BasisInterface::rparent(rpy::key_type key) const {
    return {};
}
rpy::key_type BasisInterface::index_to_key(dimn_t index) const {
    return 0;
}
dimn_t BasisInterface::key_to_index(rpy::key_type key) const {
    return 0;
}
optional<let_t> BasisInterface::first_letter(rpy::key_type key) const noexcept {
    return {};
}
optional<rpy::key_type> BasisInterface::key_of_letter(rpy::let_t letter) const noexcept {
    return {};
}
bool BasisInterface::letter(rpy::key_type key) const {
    return false;
}

optional<deg_t> Basis::width() const noexcept {
    if (p_impl) {
        return p_impl->width();
    }
    return {};
}
optional<deg_t> Basis::depth() const noexcept {
    if (p_impl) {
        return p_impl->depth();
    }
    return {};
}
optional<deg_t> Basis::degree(rpy::key_type key) const noexcept {
    if (p_impl) {
        return p_impl->degree(key);
    }
    return {};
}
std::string Basis::key_to_string(rpy::key_type key) const {
    if (p_impl) {
        return p_impl->key_to_string(key);
    }
    return {};
}
dimn_t Basis::size(int deg) const noexcept {
    if (p_impl) {
        return p_impl->size(deg);
    }
    return 0;
}
dimn_t Basis::start_of_degree(int degree) const noexcept {
    if (p_impl) {
        return p_impl->start_of_degree(degree);
    }
    return 0;
}
optional<rpy::key_type> Basis::lparent(rpy::key_type key) const {
    if (p_impl) {
        return p_impl->lparent(key);
    }
    return {};
}
optional<rpy::key_type> Basis::rparent(rpy::key_type key) const {
    if (p_impl) {
        return p_impl->rparent(key);
    }
    return {};
}
rpy::key_type Basis::index_to_key(dimn_t index) const {
    if (p_impl) {
        return p_impl->index_to_key(index);
    }
    return 0;
}
optional<let_t> Basis::first_letter(rpy::key_type key) const {
    if (p_impl) {
        return p_impl->first_letter(key);
    }
    return {};
}
optional<rpy::key_type> Basis::key_of_letter(let_t letter) const noexcept {
    if (p_impl) {
        return p_impl->key_of_letter(letter);
    }
    return {};
}
bool Basis::letter(rpy::key_type key) const {
    if (p_impl) {
        return p_impl->letter(key);
    }
    return false;
}

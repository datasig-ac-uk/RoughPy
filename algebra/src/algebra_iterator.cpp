//
// Created by user on 06/03/23.
//

#include "algebra_iterator.h"


using namespace rpy;
using namespace rpy::algebra;

const Basis &AlgebraIteratorItem::basis() const {
    assert(p_interface);
    return p_interface->basis();
}
key_type AlgebraIteratorItem::key() const noexcept {
    assert(p_interface);
    return p_interface->key();
}
scalars::Scalar AlgebraIteratorItem::value() const noexcept {
    assert(p_interface);
    return p_interface->value();
}

AlgebraIterator::AlgebraIterator(const AlgebraIterator &arg)
    : m_tag(arg.m_tag)
{
    if (arg.p_interface) {
        p_interface = arg.p_interface->clone();
    }
}
AlgebraIterator::AlgebraIterator(AlgebraIterator &&arg) noexcept
    : p_interface(std::move(arg.p_interface)), m_tag(arg.m_tag)
{
}
AlgebraIterator &AlgebraIterator::operator=(const AlgebraIterator &arg) {
    if (&arg != this) {
        if (arg.p_interface) {
            p_interface = arg.p_interface->clone();
        }
        m_tag = arg.m_tag;
    }
    return *this;
}
AlgebraIterator &AlgebraIterator::operator=(AlgebraIterator &&arg) noexcept {
    if (&arg != this) {
        p_interface = std::move(arg.p_interface);
        m_tag = arg.m_tag;
    }
    return *this;
}

AlgebraIterator &AlgebraIterator::operator++() {
    if (p_interface) {
        p_interface->advance();
    }
    return *this;
}
const AlgebraIterator AlgebraIterator::operator++(int) {
    AlgebraIterator prev(*this);
    if (p_interface) {
        p_interface->advance();
    }
    return prev;
}
AlgebraIteratorItem AlgebraIterator::operator*() const {
    return {p_interface};
}
AlgebraIteratorItem AlgebraIterator::operator->() const {
    return {p_interface};
}
bool AlgebraIterator::operator==(const AlgebraIterator &other) const {
    if (m_tag != other.m_tag) {
        return false;
    }

    if (p_interface && other.p_interface) {
        return p_interface->equals(*other.p_interface);
    }

    if (!p_interface && !other.p_interface) {
        return true;
    }

    return false;
}
bool AlgebraIterator::operator!=(const AlgebraIterator &other) const {
    return !operator==(other);
}

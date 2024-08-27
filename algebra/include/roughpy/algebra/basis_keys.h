//
// Created by sam on 8/15/24.
//

#ifndef BASIS_KEYS_H
#define BASIS_KEYS_H

#include "algebra_fwd.h"
#include <roughpy/devices/value.h>

namespace rpy {
namespace algebra {

class BasisKey : public devices::Value
{
public:
    using value_type = BasisKey;
    using const_reference_type = BasisKeyCRef;
    using reference_type = BasisKeyRef;
    using pointer_type = BasisKeyPtr;
    using const_pointer_type = BasisKeyCPtr;

    using Value::Value;

    operator const_reference_type() const noexcept;
    operator reference_type();
};

class BasisKeyCRef : public devices::ConstReference
{
public:
    using value_type = BasisKey;
    using const_reference_type = BasisKeyCRef;
    using reference_type = BasisKeyRef;
    using pointer_type = BasisKeyPtr;
    using const_pointer_type = BasisKeyCPtr;

    using ConstReference::ConstReference;

    explicit BasisKeyCRef(ConstReference&& ref) : ConstReference(std::move(ref))
    {}
};

class BasisKeyRef : public devices::Reference
{
public:
    using value_type = BasisKey;
    using const_reference_type = BasisKeyCRef;
    using reference_type = BasisKeyRef;
    using pointer_type = BasisKeyPtr;
    using const_pointer_type = BasisKeyCPtr;

    using Reference::Reference;
};

class BasisKeyCPtr : BasisKeyCRef
{
public:
    using value_type = BasisKey;
    using const_reference_type = BasisKeyCRef;
    using reference_type = BasisKeyRef;
    using pointer_type = BasisKeyPtr;
    using const_pointer_type = BasisKeyCPtr;

    using BasisKeyCRef::BasisKeyCRef;

    BasisKeyCRef operator*() const noexcept
    {
        return static_cast<BasisKeyCRef>(*this);
    }

    const BasisKeyCRef* operator->() const noexcept
    {
        return static_cast<const BasisKeyCRef*>(this);
    }
};

class BasisKeyPtr : BasisKeyRef
{
public:
    using value_type = BasisKey;
    using const_reference_type = BasisKeyCRef;
    using reference_type = BasisKeyRef;
    using pointer_type = BasisKeyPtr;
    using const_pointer_type = BasisKeyCPtr;

    using BasisKeyRef::BasisKeyRef;

    BasisKeyRef operator*() noexcept { return static_cast<BasisKeyRef>(*this); }

    BasisKeyRef* operator->() noexcept
    {
        return static_cast<BasisKeyRef*>(this);
    }
};

inline BasisKey::operator typename BasisKey::const_reference_type(
) const noexcept
{
    return const_reference_type(type(), data());
}

inline BasisKey::operator typename BasisKey::reference_type()
{
    return reference_type(type(), data());
}

}// namespace algebra
}// namespace rpy

#endif// BASIS_KEYS_H

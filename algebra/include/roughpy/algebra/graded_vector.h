//
// Created by sam on 4/19/24.
//

#ifndef ROUGHPY_ALGEBRA_GRADED_VECTOR_H
#define ROUGHPY_ALGEBRA_GRADED_VECTOR_H

#include "vector.h"

namespace rpy {
namespace algebra {

template <typename Base = Vector>
class GradedVector;

template <typename Base>
class GradedVectorContext : public Base::context_interface_t
{
    using base_t = typename Base::context_interface_t;

public:
    using base_t::base_t;

    virtual deg_t degree(const GradedVector<Base>& arg) const noexcept = 0;

    virtual GradedVector<Base>
    truncate(const GradedVector<Base>& arg, deg_t degree) const = 0;
};

/**
 * @class GradedVector
 * @brief A class that represents a graded vector.
 *
 * This class extends the Base class and provides additional functionality for
 * graded vectors.
 */
template <typename Base>
class GradedVector : public Base
{

protected:
    /**
     * @brief Resize the degree of the GradedVector object.
     *
     * This method resizes the degree of the GradedVector object to the
     * specified degree.
     *
     * @param degree The new degree to which to resize the GradedVector object.
     */
    void resize_degree(deg_t degree);

    static bool basis_compatibility_check(const Basis& basis) noexcept
    {
        return basis.is_graded() && Base::basis_compatibility_check(basis);
    }

public:
    using context_interface_t = GradedVectorContext<Base>;

private:
    const context_interface_t& get_context() const noexcept
    {
        return static_cast<const context_interface_t&>(get_context(*this));
    }

public:
    using Base::Base;

    /**
     * @brief Get the degree of the GradedVector object.
     *
     * This method returns the degree of the GradedVector object.
     *
     * @return The degree of the GradedVector object.
     */
    deg_t degree() const;

    RPY_NO_DISCARD GradedVector truncate(deg_t degree) const;
};

template <typename Base>
void GradedVector<Base>::resize_degree(deg_t degree)
{
    this->resize_dim(this->basis()->dimension_to_degree(degree));
}

template <typename Base>
deg_t GradedVector<Base>::degree() const
{
    return get_context().degree(*this);
}

template <typename Base>
GradedVector<Base> GradedVector<Base>::truncate(deg_t degree) const
{
    return get_context().truncate(*this, degree);
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_GRADED_VECTOR_H

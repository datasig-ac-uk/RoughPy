// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_SCALARS_SCALAR_INTERFACE_H_
#define ROUGHPY_SCALARS_SCALAR_INTERFACE_H_

#include "scalars_fwd.h"
#include <roughpy/platform/alloc.h>

namespace rpy {
namespace scalars {

/**
 * @class ScalarInterface
 *
 * @brief The ScalarInterface class is an interface for scalar objects.
 *
 * The ScalarInterface class provides methods for accessing and manipulating
 * scalar values. It is designed to be used as a base class for different types
 * of scalar objects.
 */
class ROUGHPY_SCALARS_EXPORT ScalarInterface : public platform::SmallObjectBase
{
public:
    virtual ~ScalarInterface();

    /**
     * @brief Returns a pointer to the constant void.
     *
     * This method returns a pointer to the constant void. It is a virtual
     * method, which means it must be implemented in derived classes. The method
     * is declared as 'const noexcept', indicating that it does not modify any
     * member variables and does not throw any exceptions.
     *
     * @return A pointer to the constant void.
     */
    RPY_NO_DISCARD virtual const void* pointer() const noexcept = 0;

    /**
     * @brief Returns the packed scalar type.
     *
     * This pure virtual function is used to get the packed scalar type of an
     * object.
     *
     * @return The packed scalar type of the object.
     *
     * @note This function does not throw any exceptions.
     */
    virtual PackedScalarType type() const noexcept = 0;

    /**
     * @brief Set the value of the object.
     *
     * This method sets the value of the object to the provided scalar value.
     * The value can be of any supported scalar type.
     *
     * @param value The scalar value to set.
     *
     * @note This method is a pure virtual function and must be overridden in
     * derived classes.
     */
    virtual void set_value(const Scalar& value) = 0;
    /**
     * @brief Prints the object to the specified output stream.
     *
     * This method is a pure virtual function defined in an abstract base class,
     * which means that it does not have an implementation in the current class
     * and should be overridden by derived classes. The purpose of this method
     * is to print the object to the specified output stream using the <<
     * operator.
     *
     * @param os The output stream to which the object will be printed.
     *
     * @see operator<<()
     */
    virtual void print(std::ostream& os) const = 0;

    /**
     * @brief Adds the provided scalar value to this object in-place.
     *
     * This method adds the provided scalar value to this object in-place,
     * modifying the value of this object.
     *
     * @param other The scalar value to add to this object.
     *
     * @note This method is defined in the ScalarInterface class and can be
     * overridden in derived classes.
     */
    virtual void add_inplace(const Scalar& other);
    /**
     * @brief Subtracts the provided scalar value from this object in-place.
     *
     * This method subtracts the provided scalar value from this object
     * in-place, modifying the value of this object.
     *
     * @param other The scalar value to subtract from this object.
     *
     * @note This method is defined in the ScalarInterface class and can be
     * overridden in derived classes.
     */
    virtual void sub_inplace(const Scalar& other);
    /**
     * @brief Multiplies the object by the provided scalar value in-place.
     *
     * This method multiplies the object by the provided scalar value in-place,
     * modifying the value of this object.
     *
     * @param other The scalar value to multiply this object with.
     *
     * @note This method is defined in the ScalarInterface class and can be
     * overridden in derived classes.
     */
    virtual void mul_inplace(const Scalar& other);
    /**
     * @brief Performs in-place division of the current scalar object by another
     * scalar.
     *
     * This method divides the value of the current scalar object by the value
     * of the given scalar. The result is stored in the current scalar object
     * itself.
     *
     * @param other The scalar object to divide by.
     *
     * @return None.
     */
    virtual void div_inplace(const Scalar& other);
};

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_INTERFACE_H_

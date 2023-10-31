// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_INTERFACE_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_INTERFACE_H_

#include "algebra_interface.h"

namespace rpy {
namespace algebra {

template <typename Bundle, typename Base, typename Fibre>
class BundleInterface : public dtl::algebra_base_resolution<
                                Bundle,
                                typename Base::basis_type,
                                dtl::AlgebraArithmetic,
                                dtl::AlgebraElementAccess>::type
{
public:
    using base_alg_t = Base;
    using fibre_alg_t = Fibre;
    using base_interface_t = typename Base::interface_t;
    using fibre_interface_t = typename Fibre::interface_t;

    virtual base_alg_t base() = 0;
    virtual fibre_alg_t fibre() = 0;
};

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_INTERFACE_H_

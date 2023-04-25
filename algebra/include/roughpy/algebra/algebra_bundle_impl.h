// Copyright (c) 2023 RoughPy Developers. All rights reserved.
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

#ifndef ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_IMPL_H_
#define ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_IMPL_H_


#include "algebra_bundle.h"
#include "algebra_impl.h"



namespace rpy {
namespace algebra {

template <typename Bundle>
struct bundle_traits;

template <typename Interface, typename BundleImpl, template <typename> class StorageModel>
class AlgebraBundleImplementation
    : protected StorageModel<BundleImpl>, public ImplAccessLayer<Interface, BundleImpl> {

    using storage_base_t = StorageModel<BundleImpl>;
    using access_layer_t = ImplAccessLayer<Interface, BundleImpl>;

    using base_alg_t = typename Interface::base_t;
    using fibre_alg_t = typename Interface::fibre_t;

    using base_interface_t = typename Interface::base_interface_t;
    using fibre_interface_t = typename Interface::fibre_interface_t;

    using bundle_traits_t = bundle_traits<BundleImpl>;

    using real_base_t = typename bundle_traits_t::base_type;
    using real_fibre_t = typename bundle_traits_t::fibre_type;

    using base_impl_t = AlgebraImplementation<base_interface_t, real_base_t, BorrowedStorageModel>;
    using fibre_impl_t = AlgebraImplementation<fibre_interface_t, real_fibre_t, BorrowedStorageModel>;


};




}
}

#endif // ROUGHPY_ALGEBRA_ALGEBRA_BUNDLE_IMPL_H_

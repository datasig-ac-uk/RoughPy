//
// Created by sammorley on 10/11/24.
//

#ifndef ROUGHPY_CORE_SMART_PTR_H
#define ROUGHPY_CORE_SMART_PTR_H

#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>

namespace rpy {


namespace mem {

template <typename T>
using Rc = boost::intrusive_ptr<T>;

template <typename T, typename Policy=boost::thread_safe_counter>
using RcBase = boost::intrusive_ref_counter<T, Policy>;



}


using mem::Rc;


}

#endif //ROUGHPY_CORE_SMART_PTR_H

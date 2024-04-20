//
// Created by sam on 4/19/24.
//

#ifndef ROUGHPY_CORE_SMART_PTR_H
#define ROUGHPY_CORE_SMART_PTR_H

#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>

namespace rpy {

template <typename T>
using Rc = boost::intrusive_ptr<T>;

template <typename Derived, typename Policy = boost::thread_safe_counter>
using RcBase = boost::intrusive_ref_counter<Derived, Policy>;

}// namespace rpy

#endif// ROUGHPY_CORE_SMART_PTR_H

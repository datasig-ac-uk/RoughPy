//
// Created by sam on 14/11/23.
//


#include "raw_bytes.h"
#include "do_macro.h"

using namespace rpy;
using namespace scalars;


namespace {

template <typename T>
std::vector<byte> to_raw_bytes_impl(const T* data, dimn_t size)
{

    return {};
}


}


std::vector<byte> scalars::dtl::to_raw_bytes(
        const void* ptr,
        dimn_t size,
        const devices::TypeInfo& info
)
{
    return {};
#define X(TP) return to_raw_bytes_impl((const TP*) ptr, size)
    DO_FOR_EACH_X(info)
#undef X
}


namespace {

template <typename T>
void from_raw_bytes(T* dst, dimn_t count, Slice<byte> bytes);

}


void scalars::dtl::from_raw_bytes(
        void* dst,
        dimn_t count,
        Slice<byte> bytes,
        const devices::TypeInfo& info
)
{
#define X(TP) return from_raw_bytes((TP*) dst, count, bytes)
DO_FOR_EACH_X(info)
#undef X
}

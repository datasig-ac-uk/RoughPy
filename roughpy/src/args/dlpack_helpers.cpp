//
// Created by sam on 11/08/23.
//

#include "dlpack_helpers.h"

using namespace rpy;
using namespace python;



#define DLTHROW(type, bits) \
    RPY_THROW(std::runtime_error, \
              std::to_string(bits) + " bit " #type " is not supported")

const string&
python::type_id_for_dl_info(const DLDataType& dtype, const DLDevice& device)
{
    if (device.device_type != kDLCPU) {
        RPY_THROW(
                std::runtime_error,
                "for the time being, constructing non-cpu "
                "devices are not supported by RoughPy"
        );
    }

    switch (dtype.code) {
        case kDLFloat:
            switch (dtype.bits) {
                case 16: return scalars::type_id_of<scalars::half>();
                case 32: return scalars::type_id_of<float>();
                case 64: return scalars::type_id_of<double>();
                default: DLTHROW(float, dtype.bits);
            }
        case kDLInt:
            switch (dtype.bits) {
                case 8: return scalars::type_id_of<char>();
                case 16: return scalars::type_id_of<short>();
                case 32: return scalars::type_id_of<int>();
                case 64: return scalars::type_id_of<long long>();
                default: DLTHROW(int, dtype.bits);
            }
        case kDLUInt:
            switch (dtype.bits) {
                case 8: return scalars::type_id_of<unsigned char>();
                case 16: return scalars::type_id_of<unsigned short>();
                case 32: return scalars::type_id_of<unsigned int>();
                case 64: return scalars::type_id_of<unsigned long long>();
                default: DLTHROW(uint, dtype.bits);
            }
        case kDLBfloat:
            if (dtype.bits == 16) {
                return scalars::type_id_of<scalars::bfloat16>();
            } else {
                DLTHROW(bfloat, dtype.bits);
            }
        case kDLComplex: DLTHROW(complex, dtype.bits);
        case kDLOpaqueHandle: DLTHROW(opaquehandle, dtype.bits);
        case kDLBool: DLTHROW(bool, dtype.bits);
    }
    RPY_UNREACHABLE_RETURN({});
}

#undef DLTHROW



const scalars::ScalarType*
python::scalar_type_for_dl_info(const DLDataType& dtype, const DLDevice& device)
{
    return scalars::ScalarType::for_id(type_id_for_dl_info(dtype, device));
}



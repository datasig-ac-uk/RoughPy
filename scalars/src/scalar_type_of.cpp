//
// Created by sam on 11/15/23.
//


#include "scalars_fwd.h"

#include <roughpy/core/check.h>        // for throw_exception, RPY_THROW

#include "scalar_type.h"
#include "scalar_types.h"

#include <roughpy/platform/devices/core.h>
#include <roughpy/platform/devices/host_device.h>


#include "scalar/do_macro.h"

using namespace rpy;
using namespace rpy::scalars;



optional<const ScalarType*> rpy::scalars::scalar_type_of(devices::TypeInfo info)
{
#define X(TP) return scalar_type_of<TP>()
    DO_FOR_EACH_X(info)
#undef X

    RPY_THROW(std::runtime_error, "unsupported scalar type");
}

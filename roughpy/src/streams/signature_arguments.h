//
// Created by sam on 11/12/24.
//

#ifndef ROUGHPY_PYMODULE_SIGNATURE_ARGUMENTS_H
#define ROUGHPY_PYMODULE_SIGNATURE_ARGUMENTS_H

#include "roughpy_module.h"

#include "roughpy/core/types.h"

#include "roughpy/streams/stream_base.h"
#include "roughpy/intervals/interval.h"
#include "roughpy/intervals/real_interval.h"

namespace rpy::python {


struct SigArgs {
    optional<intervals::RealInterval> interval;
    optional<resolution_t> resolution;
    algebra::context_pointer ctx;
};

int parse_sig_args(
        PyObject* args,
        PyObject* kwargs,
        const streams::StreamMetadata* smeta,
        SigArgs* sigargs
);

}

#endif //ROUGHPY_PYMODULE_SIGNATURE_ARGUMENTS_H

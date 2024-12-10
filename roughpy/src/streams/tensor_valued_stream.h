//
// Created by sammorley on 10/12/24.
//

#ifndef ROUGHPY_PYMODULE_TENSOR_VALUED_STREAM_H
#define ROUGHPY_PYMODULE_TENSOR_VALUED_STREAM_H


#include "roughpy_module.h"

#include "roughpy/streams/value_stream.h"
#include "roughpy/algebra/free_tensor.h"

namespace rpy::python {


void init_tensor_valued_stream(py::module_& m);


extern PyTypeObject TensorValuedStream_Type;

py::object TensorValuedStream_FromPtr(
    std::shared_ptr<const streams::ValueStream<algebra::FreeTensor>> ptr);

}


#endif //ROUGHPY_PYMODULE_TENSOR_VALUED_STREAM_H
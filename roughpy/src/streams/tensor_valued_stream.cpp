//
// Created by sammorley on 10/12/24.
//

#include "tensor_valued_stream.h"


#include "roughpy/algebra/free_tensor.h"

#include "roughpy/streams/value_stream.h"

using namespace pybind11::literals;
using namespace rpy::streams;

void rpy::python::init_tensor_valued_stream(py::module_& m)
{
    py::class_<ValueStream<algebra::FreeTensor>, std::shared_ptr<const
        ValueStream<algebra::FreeTensor>> >(m, "SimpleTensorValuedStream");







}
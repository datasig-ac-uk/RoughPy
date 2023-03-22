#include "segmentation.h"

#include <roughpy/intervals/segmentation.h>

using namespace rpy;
using namespace rpy::intervals;
using namespace pybind11::literals;


static const char* SEGMENT_DOC = R"rpydoc(Perform dyadic segmentation on an interval.
)rpydoc";

void python::init_segmentation(py::module_ &m) {

    m.def("segment",
          &segment,
          "interval"_a,
          "predicate"_a,
          "max_depth"_a,
          SEGMENT_DOC
          );
}

#include "intervals.h"


#include "dyadic.h"
#include "dyadic_interval.h"
#include "interval.h"
#include "real_interval.h"
#include "segmentation.h"

using namespace rpy;





void python::init_intervals(pybind11::module_ &m) {
    init_interval(m);
    init_real_interval(m);
    init_dyadic(m);
    init_dyadic_interval(m);
    init_segmentation(m);
}

#include "recombine.h"

#include <recombine/recombine.h>

#include "scalars/scalars.h"

using namespace rpy;
using namespace pybind11::literals;

static void recombine_wrapper(dimn_t stCubatureDimension,
                              dimn_t dimension,
                              dimn_t no_locations,
                              dimn_t *pno_kept_locations,
                              const void **ppLocationBuffer,
                              scalar_t *pdWeightBuffer,
                              idimn_t *pKeptLocations,
                              scalar_t *pNewWeights) {
    auto &no_kept_locations = *pno_kept_locations;
    auto iNoDimensionsToCubature = RdToPowersCubatureDimension(dimension, stCubatureDimension);
    if (no_locations == 0) {
        no_kept_locations = iNoDimensionsToCubature;
        return;
    }

    if (no_kept_locations < iNoDimensionsToCubature) {
        no_kept_locations = 0;
        return;
    }

    CMultiDimensionalBufferHelper sConditioning{
        dimension,
        stCubatureDimension};

    sCloud in{
        dimn_t(no_locations),
        pdWeightBuffer,
        const_cast<void *>(static_cast<const void *>(ppLocationBuffer)),
        &sConditioning};

    sRCloudInfo out{
        iNoDimensionsToCubature,
        pNewWeights,
        (dimn_t *)pKeptLocations,
        nullptr};

    sRecombineInterface data{
        &in,
        &out,
        iNoDimensionsToCubature,
        &RdToPowers,
        nullptr};

    Recombine(&data);

    no_kept_locations = data.pOutCloudInfo->No_KeptLocations;
}

static py::tuple py_recombine(const py::object& data,
                              const py::object& src_locs,
                              const py::object& src_weights,
                              deg_t degree) {

    python::PyToBufferOptions options;
    options.max_nested = 2;
    options.allow_scalar = false;
    std::vector<double> data_bk;
    std::vector<double> src_weights_bk;
    std::vector<std::size_t> src_locs_bk;

    auto buffer = python::py_to_buffer(data, options);

    const auto& shape = options.shape;
    const auto ndim = shape.size();

    if (ndim < 2 || buffer.size() == 0) {
        throw py::value_error("malformed data");
    }

    auto no_data_points = shape[0];
    auto point_dimension = shape[1];



}


void python::init_recombine(pybind11::module_ &m) {

    m.def("recombine", &py_recombine,
          "data"_a,
          "src_locations"_a=py::none(),
          "src_weights"_a=py::none(),
          "degree"_a=1
          );




}

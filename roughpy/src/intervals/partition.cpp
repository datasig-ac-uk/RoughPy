// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 05/07/23.
//

#include "partition.h"

#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <algorithm>

using namespace rpy;
using namespace intervals;
using namespace pybind11::literals;

PyDoc_VAR(PyPartition_doc) = R"rpydoc(
An :class:`Interval` into which there are a number of intermediate points which represent the end points of sub intervals.
Partition of an :class:`Interval` in the real line.
)rpydoc";

namespace {

Partition partition_py_ctor(const RealInterval& interval,
                            const py::iterable& py_intermediates)
{
    const auto inf = interval.inf();
    const auto sup = interval.sup();
    std::vector<param_t> intermediates;
    for (auto&& mid : py_intermediates) {
        auto param = mid.cast<param_t>();
        if (interval.contains_point(param) && param != inf && param != sup) {
            intermediates.push_back(param);
        }
    }

    std::sort(intermediates.begin(), intermediates.end());
    return Partition(interval, std::move(intermediates));
}

}// namespace

void python::init_partition(py::module_& m)
{

    py::class_<Partition, Interval> cls(m, "Partition", PyPartition_doc);

    cls.def(py::init<RealInterval>(), "interval"_a);
    cls.def(py::init(&partition_py_ctor), "interval"_a, "intermediates"_a);

    cls.def("__len__", &Partition::size);
    cls.def("__getitem__", &Partition::operator[], "index"_a);
    cls.def("refine_midpoints", &Partition::refine_midpoints, "Inserts the midpoint between all of the intermediates.");
    cls.def("mesh", &Partition::mesh, "Length of the largest sub interval.");
    cls.def("intermediates", &Partition::intermediates, "List of intermediate end points.");
    cls.def("insert_intermediate", &Partition::insert_intermediate, "Cuts one of the intermediate intervals in half.");
    cls.def("merge", &Partition::merge, "The union of the partitions.");

    cls.def("__str__", [](const Partition& self) {
        std::stringstream ss;
        for (dimn_t i=0; i<self.size(); ++i) {
            ss << self[i];
        }
        return ss.str();
    });

    cls.def(py::pickle(
            [](const Partition& value) -> py::tuple {
                return py::make_tuple(
                        value.type(),
                        value.inf(),
                        value.sup(),
                        value.intermediates()
                        );
            },
            [](py::tuple state) -> Partition {
                if (state.size() != 4) {
                    throw std::runtime_error("invalid state");
                }

                return Partition(
                        RealInterval(state[1].cast<param_t>(),
                        state[2].cast<param_t>(),
                        state[0].cast<intervals::IntervalType>()),
                        state[3].cast<typename Partition::intermediates_t>()
                        );
            }
            ));


    cls.def(py::self == py::self);
}

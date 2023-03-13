//
// Created by sam on 13/03/23.
//

#include "ContextFixture.h"


using namespace rpy::algebra;

rpy::algebra::testing::ContextFixture::ContextFixture() {
    stype = scalars::ScalarType::of<double>();
    ctx = get_context(width, depth, stype, {{"backend", "libalgebra_lite"}});
}

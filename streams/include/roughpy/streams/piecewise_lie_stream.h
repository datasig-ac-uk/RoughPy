#ifndef ROUGHPY_STREAMS_PIECEWISE_LIE_STREAM_H_
#define ROUGHPY_STREAMS_PIECEWISE_LIE_STREAM_H_


#include "stream_base.h"

#include <roughpy/config/implementation_types.h>
#include <roughpy/algebra/lie.h>
#include <roughpy/algebra/free_tensor.h>
#include <roughpy/algebra/context.h>

namespace rpy { namespace streams {


class PiecewiseLiePath : public StreamInterface {
public:
    using LiePiece = std::pair<intervals::RealInterval, algebra::Lie>;

private:

    std::vector<LiePiece> m_data;

    static inline param_t to_proportion(const intervals::RealInterval& interval, param_t param) {
        assert(interval.inf() <= param && param <= interval.sup());
        return (param - interval.inf()) / (interval.sup() - interval.inf());
    }

};


}}


#endif // ROUGHPY_STREAMS_PIECEWISE_LIE_STREAM_H_

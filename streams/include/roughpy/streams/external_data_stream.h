#ifndef ROUGHPY_STREAMS_EXTERNAL_DATA_STREAM_H_
#define ROUGHPY_STREAMS_EXTERNAL_DATA_STREAM_H_

#include "stream_base.h"

#include <memory>

namespace rpy { namespace streams {


class ROUGHPY_STREAMS_EXPORT ExternalDataStreamSource {

public:

    virtual ~ExternalDataStreamSource();

    virtual scalars::KeyScalarArray query(const intervals::Interval& interval) = 0;

};


class ROUGHPY_STREAMS_EXPORT ExternalDataStream : public StreamInterface {
    std::unique_ptr<ExternalDataStreamSource> p_source;

public:




};


}}

#endif // ROUGHPY_STREAMS_EXTERNAL_DATA_STREAM_H_

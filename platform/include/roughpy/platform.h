//
// Created by sam on 14/04/23.
//

#ifndef ROUGHPY_PLATFORM_H
#define ROUGHPY_PLATFORM_H

#include <roughpy/core/helpers.h>

#include <boost/dll/shared_library.hpp>
#include <boost/url.hpp>
#include <boost/url/parse.hpp>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/string_generator.hpp>
#include <boost/uuid/name_generator.hpp>
#include <boost/uuid/uuid_io.hpp>


#include "platform/configuration.h"
#include "platform/filesystem.h"

namespace rpy {

using boost::url;
using boost::url_view;

using boost::urls::parse_uri;
using boost::urls::parse_uri_reference;
using URIScheme = boost::urls::scheme;

using boost::dll::shared_library;


namespace uuids = boost::uuids;

}// namespace rpy

#endif// ROUGHPY_PLATFORM_H

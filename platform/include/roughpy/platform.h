//
// Created by sam on 14/04/23.
//

#ifndef ROUGHPY_PLATFORM_H
#define ROUGHPY_PLATFORM_H

#include <roughpy/core/helpers.h>

#include <boost/url.hpp>
#include <boost/url/parse.hpp>

#include <boost/dll/shared_library.hpp>

#include "platform/filesystem.h"
#include "platform/configuration.h"

namespace rpy {

using boost::url;
using boost::url_view;

using boost::urls::parse_uri;
using boost::urls::parse_uri_reference;
using URIScheme = boost::urls::scheme;


using boost::dll::shared_library;


}



#endif//ROUGHPY_PLATFORM_H

#ifndef ROUGHPY_PLATFORM_FILESYSTEM_H_
#define ROUGHPY_PLATFORM_FILESYSTEM_H_

#ifdef RPY_CPP_17
#  include <filesystem>
#else
#  include <boost/filesystem.hpp>
#endif

namespace rpy {
namespace fs {

#if defined(RPY_CPP_17)
using std::filesystem::path;

using std::filesystem::absolute;
using std::filesystem::canonical;
using std::filesystem::copy;
using std::filesystem::create_directory;
using std::filesystem::current_path;
using std::filesystem::equivalent;
using std::filesystem::exists;
using std::filesystem::file_size;
using std::filesystem::is_directory;
using std::filesystem::is_regular_file;
using std::filesystem::permissions;
using std::filesystem::remove;
using std::filesystem::rename;

#else
using boost::filesystem::path;

using boost::filesystem::absolute;
using boost::filesystem::canonical;
using boost::filesystem::copy;
using boost::filesystem::create_directory;
using boost::filesystem::current_path;
using boost::filesystem::equivalent;
using boost::filesystem::exists;
using boost::filesystem::file_size;
using boost::filesystem::is_directory;
using boost::filesystem::is_regular_file;
using boost::filesystem::permissions;
using boost::filesystem::remove;
using boost::filesystem::rename;

#endif
}// namespace fs
}// namespace rpy

#endif// ROUGHPY_PLATFORM_FILESYSTEM_H_

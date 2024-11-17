//
// Created by sammorley on 08/11/24.
//

#include <gtest/gtest.h>

#include <cstdio>

#if defined(_WIN32) && defined(_DEBUG) && !defined(RPY_NO_DEBUG_MODIFICATION)
#  include <crtdbg.h>

#define(RPY_SET_REPORT_FILE, TP, FL)                                           \
    do {                                                                       \
        err_code = _CrtSetReportMode((TP), (FL));                              \
        if (err_code != 0) {                                                   \
            return err_code;                                                   \
        }                                                                      \
    } while (0)

#endif

/*
 * This is a slight modification of the main function found in the gtest_main.cc
 * file of gtest library. This version adds calls to change the default handlers
 * for debug assertions and errors. With this configuration, errors are dumped
 * to stderr rather than using a debug popup, which for some reason is the
 * default
 */
int main(int argc, char** argv)
{

#if defined(_WIN32) && defined(_DEBUG) && !defined(RPY_NO_DEBUG_MODIFICATION)
    int err_code = 0;
    int warn_mode = _CrtSetReportMode(_CRT_WARN,
        _CRTDBG_MODE_DEBUG | _CRTDBG_MODE_FILE);
    RPY_SET_REPORT_FILE(_CRT_WARN, stderr);
    int err_mode = _CrtSetReportMode(_CRT_ERROR,
        _CRTDBG_MODE_DEBUG | _CRTDBG_MODE_FILE);
    RPY_SET_REPORT_FILE(_CRT_ERROR, stderr);
    _CrtSetReportFile(_CRT_ERROR, stderr);
    int assert_mode = _CrtSetReportMode(_CRT_ASSERT,
        _CRTDBG_MODE_DEBUG | _CRTDBG_MODE_FILE);
    RPY_SET_REPORT_FILE(_CRT_ASSERT, stderr);
#endif

    // make sure the output format is exactly as for the gtest_main function
    printf("Running main() from file %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    const auto result = RUN_ALL_TESTS();

#if defined(_WIN32) && defined(_DEBUG) && !defined(RPY_NO_DEBUG_MODIFICATION)
    // I don't know if resetting these modes is strictly necessary, but it
    // probably doesn't hurt,
    _CrtSetReportMode(_CRT_WARN, warn_mode);
    _CrtSetReportMode(_CRT_ERROR, error_mode);
    _CrtSetReportMode(_CRT_ASSERT, assert_mode);
#endif
    return result;
}
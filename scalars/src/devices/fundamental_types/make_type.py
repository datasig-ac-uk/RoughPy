import re
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).parent


def generate_h_file(type_names):
    with open('extern_vars.h', 'w') as f:
        # include guard
        f.write("#ifndef EXTERN_VARS_H\n#define EXTERN_VARS_H\n\n")
        # include the headers for FundamentalType and the types being used
        f.write("#include \"FundamentalType.h\"\n\n")
        for type_name in type_names:
            # extern variable declaration for each type
            f.write(
                f"extern FundamentalType<{type_name}> extern_var_{type_name};\n")
        f.write("\n#endif // EXTERN_VARS_H\n")


def generate_cpp_file(type_names):
    with open('extern_vars.cpp', 'w') as f:
        # include the header file
        f.write('#include "extern_vars.h"\n\n')
        for type_name in type_names:
            # extern variable definition for each type
            f.write(f"FundamentalType<{type_name}> extern_var_{type_name};\n")


def write_file(type_name: str):
    if (match := re.match(r"(u)?int(\d+)_t", type_name)) is not None:
        name = type_name[:-2]
        display_name = name
        type_id = f"{match.group(1) or 'i'}{match.group(2)}"
    elif (match := re.match(r"((\w)\w+)(?:_t)?", type_name)) is not None:
        name = match.group(1)
        type_id = name
        display_name = name
    else:
        raise RuntimeError(f"bad type name {type_name}")

    filename = f"{name}_type"
    var_name = f"{name}_type"
    file = ROOT_PATH / filename
    print(f"Writing {file}")

    guard_name = f"{filename.upper()}_H_"
    with open(file.with_suffix(".h"), "wt") as fp:
        fp.write("\n".join([
            f"#ifndef {guard_name}",
            f"#define {guard_name}",
            "#include \"devices/fundamental_type.h\"",
            "#include <roughpy/core/types.h>",
            "#include <roughpy/core/macros.h>",
            "namespace rpy {",
            "namespace devices {",
            f"extern template class RPY_LOCAL FundamentalType<{type_name}>;",
            f"extern RPY_LOCAL const FundamentalType<{type_name}> {var_name};",
            "}",
            "}",
            f"#endif // {guard_name}"
        ]))

    with open(file.with_suffix(".cpp"), "wt") as fp:
        fp.write("\n".join([
            f"#include \"{file.with_suffix('.h').name}\"",
            "namespace rpy {",
            "namespace devices {",
            f"template class FundamentalType<{type_name}>;",
            "}",
            "}",
            "using namespace rpy;",
            "using namespace rpy::devices;",
            f"const FundamentalType<{type_name}>",
            f"    devices::{var_name}(\"{type_id}\", \"{display_name}\");",
        ]))


def main(argv):
    for tp in argv[1:]:
        write_file(tp)


if __name__ == "__main__":
    main(sys.argv)

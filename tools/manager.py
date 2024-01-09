
import argparse
import sys
import re

from argparse import ArgumentParser
from pathlib import Path

import semver


ROUGHPY_ROOT = Path(__file__).parent.parent
assert (ROUGHPY_ROOT / "CMakeLists.txt").exists()


def get_current_version() -> semver.Version:
    path = ROUGHPY_ROOT / "VERSION.txt"
    assert path.is_file()

    version_text = path.read_text()
    if version_text.startswith("v"):
        version_text = version_text[1:]

    return semver.Version.parse(version_text)


def format_version(version: semver.Version) -> str:
    return f"v{version.major}.{version.minor}.{version.patch}"


def write_version(version: semver.Version):
    path = ROUGHPY_ROOT / "VERSION.txt"
    path.write_text(format_version(version))


def bump_version(bump_type):
    version = get_current_version()
    new_version = getattr(version, f"bump_{bump_type}")()
    print("New version:", new_version)
    write_version(new_version)



def do_release(args):

    bump_version(args.bump_type)




def main():
    parser = ArgumentParser("manager.py")

    subparsers = parser.add_subparsers(
        title="Actions",
        description="The following actions are available",
        required=True,
        metavar=None
    )

    update_version_a = subparsers.add_parser(
        "update-version",
        help="Update the version file to contain the current version"
    )
    update_version_a.add_argument("new_version")

    bump_version_a = subparsers.add_parser(
        "release",
            help="Bump the current version and make a new release"
    )
    version_bump_type_gp = bump_version_a.add_mutually_exclusive_group()
    version_bump_type_gp.add_argument(
        "--major",
        dest="bump_type",
        action="store_const",
        const="major"
    )
    version_bump_type_gp.add_argument(
        "--minor",
        dest="bump_type",
        action="store_const",
        const="minor"
    )
    version_bump_type_gp.add_argument(
        "--patch",
        dest="bump_type",
        action="store_const",
        const="patch"
    )
    version_bump_type_gp.set_defaults(bump_type="patch")

    bump_version_a.set_defaults(func=do_release)

    args = parser.parse_args()
    args.func(args)

    parser.exit(0)



if __name__ == "__main__":
    main()
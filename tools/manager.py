
import argparse
import sys
import re

from argparse import ArgumentParser
from pathlib import Path




def do_update_version(args):
    print("update version to", args.new_version)


def do_release(args):
    pass



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
    update_version_a.set_defaults(func=do_update_version)

    bump_version_a = subparsers.add_parser(
        "release",
            help="Bump the current version and make a new release"
    )
    version_bump_type_gp = bump_version_a.add_mutually_exclusive_group()
    version_bump_type_gp.add_argument(
        "--major",
        metavar="bump_type",
        action="store_const",
        const="major"
    )
    bump_version_a.set_defaults(func=do_release)



    args = parser.parse_args()
    args.func(args)

    parser.exit(0)



if __name__ == "__main__":
    main()
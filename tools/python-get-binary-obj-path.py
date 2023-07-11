#  Copyright (c) 2023 Datasig Developers. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification,
#  are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#  may be used to endorse or promote products derived from this software without
#  specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
#  OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
#  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from __future__ import annotations

import importlib.metadata as ilm
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Generator, Iterable

CMAKE_LIST_SEP = ';'

Matcher = Callable[[ilm.PackagePath], bool]


def find_component(package: str, matcher: Matcher) -> Generator[Path]:
    try:
        dist = ilm.distribution(package)
        yield from (f.locate().resolve() for f in dist.files
                    if matcher(f))
    except ilm.PackageNotFoundError:
        pass


def _flatten(gens: Iterable[Generator[Path]]) -> Generator[Path]:
    for gen in gens:
        yield from gen


def _trim_to_path_dir(path: Path, fragment: Path) -> Path:
    prefix = path
    for l, _ in zip(path.parents, fragment.parents):
        prefix = l

    return prefix


def _trim_to_directory(paths: Generator[Path], pat: Path) -> Generator[Path]:
    for p in paths:
        yield _trim_to_path_dir(p, pat)


def make_name_matcher(name: str) -> Matcher:
    def matcher(path: ilm.PackagePath) -> bool:
        return path.stem == name

    return matcher


def make_fragment_matcher(fragment: str) -> Matcher:
    def matcher(path: ilm.PackagePath) -> bool:
        return path.match(fragment)

    return matcher


def make_matcher(pattern: str, match_name: bool) -> Matcher:
    if match_name:
        return make_name_matcher(pattern)

    return make_fragment_matcher(pattern)


def trim_count(count: int, paths: Generator[Path]) -> Generator[Path]:
    for _, p in zip(range(count), paths):
        yield p


def main():
    parser = ArgumentParser()

    parser.add_argument("package", type=str)
    parser.add_argument("pattern", nargs="+")
    parser.add_argument(
        "-e", "--exitcode",
        action="store_true",
        help="Return exit code on failure"
    )
    parser.add_argument(
        "-n", "--name",
        action="store_true",
        help="Find by file name"
    )
    parser.add_argument(
        "-d", "--directory",
        action="store_true",
        help="Find the containing directory"
    )
    parser.add_argument(
        "-c", "--count",
        type=int,
        default=0,
        nargs=1,
        required=False,
        help="Limit the number of entries returned"
    )

    args = parser.parse_args()

    paths = []

    for pat in args.pattern:
        matcher = make_matcher(pat, args.name)
        found = find_component(args.package, matcher)
        if not found and args.exitcode:
            parser.exit(1)

        if args.count > 0:
            found = trim_count(args.count, found)

        if args.directory:
            found = _trim_to_directory(found, Path(pat))

        paths.extend(found)

    if not paths and args.exitcode:
        parser.exit(1)

    print(CMAKE_LIST_SEP.join(map(str, paths)))


if __name__ == "__main__":
    main()

"""
This is an example of how to compute signatures of many paths in parallel using
the current version of the library.

Please be aware that at some later time, we will include proper tools for doing
such computations within RoughPy itself, and this technique should only be used
until a more robust solution has been constructed.

The main problem for multiprocessing is that, at present, RoughPy streams and
other objects cannot be pickled (serialized) to be passed to other processes,
so we can't simply use the multiprocessing tools naively. However, there are no
problems providing that RoughPy objects are created, manipulated, and processed
only within the process in which they are created - they cannot be shared
between processes and will cease to exist once the process finishes its work.

Currently, RoughPy does not release the Python Global Interpreter Lock (GIL)
during calculations, so thread-level parallelism won't yield any benefits,
although, long term, this is certainly a better model that multiprocessing.
"""
from __future__ import annotations


# I'm using concurrent.futures because it is part of the standard library, and
# doesn't require installing additional libraries. Using joblib or similar
# packages will need a small amount of modification.
from concurrent.futures import ProcessPoolExecutor

# we can't pass RoughPy intervals between processes either, so we'll use a
# named tuple as a stand-in
from typing import NamedTuple

from functools import partial
from pathlib import Path


import numpy as np
import roughpy as rp


class RealInterval(NamedTuple):
    inf: float
    sup: float

    @staticmethod
    def from_rp(interval: rp.Interval) -> RealInterval:
        return RealInterval(interval.inf(), interval.sup())

    def to_rp(self) -> rp.RealInterval:
        return rp.RealInterval(self.inf, self.sup)


def process_worker(file: str, intervals: list[RealInterval], resolution: int) \
        -> tuple[str, list[np.ndarray]]:
    stream = rp.ExternalDataStream.from_uri(file, width=2, depth=3,
                                            dtype=rp.DPReal)

    # We can't pass a rp.FreeTensor back from the process, so we have to convert
    # it to a NumPy array that can be sent back.
    return file, [
        np.array(stream.signature(ivl.to_rp(), resolution)) for ivl in intervals
    ]



# We're going to use the Dale-UK data set for this example
# (https://doi.org/10.5286/UKERC.EDC.000001)
# If some sample data is not already in the data folder in this directory, then
# you will need to download some from the UKRI pages linked above. You do not
# need to download the entire data.
# WARNING: This is a large data set
data_dir = Path(__file__).parent / "data"

if not data_dir.exists():
    data_dir.mkdir()

# For demonstrating the technique, we have 7 files from the following URL:
# https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2015/UK-DALE-16kHz/house_1/2013/wk15
# selected using the file spec "vi-13653*.flac" which gives us ~1GB data in 7
# files. Names are as follows:
# vi-1365375600_222578.flac		202,065,930
# vi-1365379200_763408.flac		200,426,295
# vi-1365382800_088912.flac		200,437,825
# vi-1365386400_330756.flac		200,577,979
# vi-1365390000_530581.flac		200,532,049
# vi-1365393600_367075.flac		200,275,046
# vi-1365397200_181702.flac		200,478,749
Dale_files = [str(file) for file in data_dir.iterdir()]


# We don't want to launch more processes than absolutely necessary - they are
# expensive. For now, we're going to launch at most 4 processes, only if there
# are at least 5 streams per process.
num_processes = min(len(Dale_files) // 5, 3) + 1

# A modest resolution for our computations
RESOLUTION = 5

# We want to compute a fair number of intervals per stream. Here we're computing
# signatures over 5 minute intervals over the hour contained in each file.
INTERVALS = [
    RealInterval(300.*i, 300.*(i+1)) for i in range(12)
]


with ProcessPoolExecutor(max_workers=num_processes) as pool:
    func = partial(process_worker, intervals=INTERVALS, resolution=RESOLUTION)
    results = {
        fname: sigs for fname, sigs in pool.map(func, Dale_files)
    }


for fname, sigs in results.items():
    print(f"{fname} signatures:")
    for sig in sigs:
        print(sig[:3])

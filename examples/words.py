import numpy as np

import roughpy as rp

# This should work on Debian and Ubuntu without any additional steps. On other distributions, such as Arch Linux, you
# may need to install an additional package (on Arch, the "words" package).
with open("/usr/share/dict/words", "rt") as fd:
    word_list = {word.lower() for line in fd
                 if len(word := line.strip().replace('-', '')) > 1
                 if word.isalpha() and word.isascii()}

CTX = rp.get_context(width=26, depth=3, coeffs=rp.Rational)


def word_to_stream(word):
    incr_array = np.zeros((len(word), 26), dtype=np.int8)
    for i, letter in enumerate(word):
        assert 97 <= ord(letter) <= 122, f"{letter} is not allowed"
        incr_array[i, ord(letter) - 97] = 1

    return rp.LieIncrementStream.from_increments(incr_array, resolution=2, ctx=CTX)


print(f"There are {len(word_list)} words")

word_streams = {word: word_to_stream(word) for word in word_list}


from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from time import time


def compute(word_stream, *, depth):
    """"Helper function for to get (key, word) results"""
    return str(word_stream[1].log_signature(depth=depth)), word_stream[0]


start = time()
anagrams = defaultdict(list)
with ThreadPoolExecutor(max_workers=8) as pool:
    for key, word in pool.map(partial(compute, depth=1), word_streams.items()):
        anagrams[key].append(word)

elapsed = time() - start

print(f"Computation took {elapsed} seconds")

for key, words in anagrams.items():
    if len(words) == 1:
        word_streams.pop(words[0])
    if len(words) > 6:
        print(f"{key:<40}", *words)




print(f"There are {len(word_streams)} words with at least one anagram")

start = time()
anagrams2 = defaultdict(list)
with ThreadPoolExecutor() as pool:
    for key, word in pool.map(partial(compute, depth=2), word_streams.items()):
        anagrams2[key].append(word)

elapsed = time() - start

print(f"Computation took {elapsed} seconds")
for key, words in anagrams2.items():
    if len(words) == 1:
        word_streams.pop(words[0])
    else:
        print(f"{key:<40}", *words)

print(f"There are {len(word_streams)} words whose level 3 signatures are necessary")


def compute_3(word_stream):
    return str(word_stream[1].signature()), word_stream[0]


start = time()
anagrams3 = defaultdict(list)
with ThreadPoolExecutor() as pool:
    for key, word in pool.map(compute_3, word_streams.items()):
        anagrams3[key].append(word)

elapsed = time() - start
print(f"Computation took {elapsed} seconds")

for key, words in anagrams3.items():
    if len(words) == 1:
        word_streams.pop(words[0])

print(f"There are {len(word_streams)} words whose level 4 signatures are necessary")
